from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import os
import json
import logging
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# AWS Credentials and Configurations
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", "ap-southeast-1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
DYNAMO_TABLE_NAME = os.getenv("DYNAMO_TABLE_NAME", "dev-incidents")

# Initialize FastAPI app
app = FastAPI()

# Initialize AWS clients
dynamodb_client = boto3.client(
    "dynamodb",
    region_name=DYNAMODB_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Request model
class QueryRequest(BaseModel):
    user_question: str
    company_id: str

def extract_json_from_text(text):
    """Extracts JSON content from a text response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

def generate_dynamodb_query(user_query: str, company_id: str):
    try:
        prompt = f"""
        Convert the following user question into a valid DynamoDB query.

        **Rules:**
        1. Query incidents from `{DYNAMO_TABLE_NAME}`.
        2. Always filter by `company_id`.
        3. If the user asks about a specific incident (e.g., ID, status, priority, service), include those filters.
        4. Use `query` when filtering by ID, else use `scan`.

        **Examples:**
        Example 1:
        User query: "Show me all incidents"
        Output: {{"TableName": "{DYNAMO_TABLE_NAME}", "FilterExpression": "company_id = :company_id", "ExpressionAttributeValues": {{":company_id": {{"S": "COMPANY_ID"}}}}}}
        
        Example 2: 
        User query: "Show critical incidents"
        Output: {{"TableName": "{DYNAMO_TABLE_NAME}", "FilterExpression": "company_id = :company_id AND priority = :priority", "ExpressionAttributeValues": {{":company_id": {{"S": "COMPANY_ID"}}, ":priority": {{"S": "critical"}}}}}}
        
        Example 3:
        User query: "Show incident ID-123"
        Output: {{"TableName": "{DYNAMO_TABLE_NAME}", "KeyConditionExpression": "id = :id", "ExpressionAttributeValues": {{":id": {{"S": "ID-123"}}, ":company_id": {{"S": "COMPANY_ID"}}}}}}
        
        **User Query:**  
        {user_query}

        **Company ID:**  
        {company_id}

        **Expected Output:**  
        A valid DynamoDB JSON query ONLY. No explanations.
        """

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": 500,
            "temperature": 0.1  # Lower temperature for more consistent query generation
        }

        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps(payload)
        )

        body_content = response["body"].read().decode("utf-8").strip()
        logger.info(f"Bedrock Response: {body_content}")

        if not body_content:
            return None

        response_body = json.loads(body_content)
        query_text = response_body.get("content", [{}])[0].get("text", "").strip()
        json_query = extract_json_from_text(query_text)

        if json_query:
            dynamo_query = json.loads(json_query)
            dynamo_query["TableName"] = DYNAMO_TABLE_NAME  # Ensure TableName is set

            # Ensure company_id filtering
            if "KeyConditionExpression" in dynamo_query:
                # For Key condition queries, add company_id as filter expression
                if "FilterExpression" in dynamo_query:
                    dynamo_query["FilterExpression"] += " AND company_id = :company_id"
                else:
                    dynamo_query["FilterExpression"] = "company_id = :company_id"
            elif "FilterExpression" in dynamo_query:
                if ":company_id" not in dynamo_query.get("ExpressionAttributeValues", {}):
                    dynamo_query["FilterExpression"] += " AND company_id = :company_id"
            else:
                dynamo_query["FilterExpression"] = "company_id = :company_id"

            if "ExpressionAttributeValues" not in dynamo_query:
                dynamo_query["ExpressionAttributeValues"] = {}

            if ":company_id" not in dynamo_query["ExpressionAttributeValues"]:
                dynamo_query["ExpressionAttributeValues"][":company_id"] = {"S": company_id}
            else:
                # Ensure company_id value is the actual company_id (not a placeholder)
                dynamo_query["ExpressionAttributeValues"][":company_id"] = {"S": company_id}

            return dynamo_query

        return None
    except Exception as e:
        logger.error(f"Error generating DynamoDB query: {str(e)}")
        return None

def execute_dynamodb_query(query: dict):
    try:
        if not query:
            return None

        query["TableName"] = DYNAMO_TABLE_NAME  # Ensure TableName is always included

        # Log final query
        logger.info(f"Executing Query: {json.dumps(query, indent=2)}")

        # Use 'query' if there's a KeyConditionExpression, otherwise use 'scan'
        response = (
            dynamodb_client.query(**query)
            if "KeyConditionExpression" in query
            else dynamodb_client.scan(**query)
        )

        logger.info(f"DynamoDB Response Summary: Items count = {len(response.get('Items', []))}")
        
        # Only log first item as sample if available
        if response.get('Items') and len(response.get('Items')) > 0:
            logger.info(f"Sample Item: {json.dumps(response['Items'][0], indent=2)}")

        return response.get("Items", [])

    except Exception as e:
        logger.error(f"Error executing DynamoDB query: {str(e)}")
        return None

def format_response_for_question(user_question, incidents):
    """
    Analyzes the user's question and formats a response that directly addresses it
    based on the incident data available.
    """
    # Convert question to lowercase for easier matching
    question_lower = user_question.lower()
    
    # No incidents found
    if not incidents or len(incidents) == 0:
        return {"message": "No incidents found matching your criteria."}
    
    # Handle time-to-resolution questions
    if any(keyword in question_lower for keyword in ["time to resolve", "resolution time", "how long", "how much time"]):
        incident = incidents[0]  # Take the first incident if multiple returned
        incident_id = incident.get("id", {}).get("S", "N/A")
        status = incident.get("status", {}).get("S", "Unknown").lower()
        
        # Get resolution time if available
        created_time = incident.get("created_at", {}).get("S", None)
        resolved_time = incident.get("resolved_at", {}).get("S", None)
        
        if status == "resolved" and created_time and resolved_time:
            try:
                # Parse timestamps
                from datetime import datetime
                created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                resolved_dt = datetime.fromisoformat(resolved_time.replace('Z', '+00:00'))
                
                # Calculate resolution time
                resolution_time = resolved_dt - created_dt
                hours = resolution_time.total_seconds() / 3600
                
                return {
                    "message": f"Incident {incident_id} was resolved in {hours:.2f} hours.",
                    "resolution_time_hours": round(hours, 2),
                    "incident_details": {
                        "id": incident_id,
                        "status": status,
                        "priority": incident.get("priority", {}).get("S", "Normal"),
                        "created_at": created_time,
                        "resolved_at": resolved_time
                    }
                }
            except Exception as e:
                logger.error(f"Error calculating resolution time: {str(e)}")
        
        if status != "resolved":
            return {
                "message": f"Incident {incident_id} is still {status} and has not been resolved yet.",
                "incident_details": {
                    "id": incident_id,
                    "status": status,
                    "priority": incident.get("priority", {}).get("S", "Normal"),
                    "created_at": created_time
                }
            }
        
        return {
            "message": f"Resolution time information is not available for incident {incident_id}.",
            "incident_details": {
                "id": incident_id,
                "status": status,
                "priority": incident.get("priority", {}).get("S", "Normal")
            }
        }
    
    # Handle status questions
    elif any(keyword in question_lower for keyword in ["status", "state", "condition"]):
        incident = incidents[0]
        incident_id = incident.get("id", {}).get("S", "N/A")
        status = incident.get("status", {}).get("S", "Unknown")
        
        return {
            "message": f"Incident {incident_id} is currently in {status} status.",
            "incident_details": {
                "id": incident_id,
                "status": status,
                "priority": incident.get("priority", {}).get("S", "Normal")
            }
        }
    
    # Handle priority questions
    elif any(keyword in question_lower for keyword in ["priority", "severity", "importance"]):
        incident = incidents[0]
        incident_id = incident.get("id", {}).get("S", "N/A")
        priority = incident.get("priority", {}).get("S", "Normal")
        
        return {
            "message": f"Incident {incident_id} has {priority} priority.",
            "incident_details": {
                "id": incident_id,
                "priority": priority,
                "status": incident.get("status", {}).get("S", "Unknown")
            }
        }
    
    # Default response for other queries - list found incidents
    formatted_incidents = []
    for incident in incidents:
        incident_id = incident.get("id", {}).get("S", "N/A")
        status = incident.get("status", {}).get("S", "Unknown")
        priority = incident.get("priority", {}).get("S", "Normal")
        service = incident.get("service", {}).get("S", "N/A")
        
        formatted_incidents.append({
            "id": incident_id,
            "status": status,
            "priority": priority,
            "service": service
        })
    
    return {
        "message": f"Found {len(formatted_incidents)} incident(s).",
        "incidents": formatted_incidents
    }

@app.post("/query")
async def query_dynamodb(request: QueryRequest):
    """API endpoint to process user query and fetch results from DynamoDB."""
    try:
        # Validate inputs
        if not request.company_id:
            raise HTTPException(status_code=400, detail="company_id is required")
        
        if not request.user_question:
            raise HTTPException(status_code=400, detail="user_question is required")
            
        # Generate the query from user input
        dynamo_query = generate_dynamodb_query(request.user_question, request.company_id)
        if not dynamo_query:
            raise HTTPException(status_code=500, detail="Failed to generate DynamoDB query")

        logger.info(f"Generated DynamoDB Query: {json.dumps(dynamo_query, indent=2)}")

        # Execute the query
        incidents = execute_dynamodb_query(dynamo_query)
        
        # Debug in logs but don't return to user
        if not incidents or len(incidents) == 0:
            debug_result = debug_dynamo_response(incidents, request.company_id)
            logger.info(f"Debug Result: {json.dumps(debug_result, indent=2)}")
            return {"message": "No incidents found."}
        
        # Format response based on the question type
        response = format_response_for_question(request.user_question, incidents)
        return response
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def debug_dynamo_response(incidents, company_id):
    """
    Debug function to analyze DynamoDB response and identify potential issues
    """
    logger.info(f"Debug: Analyzing response for company_id: {company_id}")
    
    if not incidents or len(incidents) == 0:
        logger.info("Debug: No incidents returned from DynamoDB")
        
        # Test direct query to verify table access
        try:
            # Simple scan with just the company_id filter
            test_query = {
                "TableName": DYNAMO_TABLE_NAME,
                "FilterExpression": "company_id = :company_id",
                "ExpressionAttributeValues": {
                    ":company_id": {"S": company_id}
                }
            }
            
            logger.info(f"Debug: Executing test query: {json.dumps(test_query)}")
            test_response = dynamodb_client.scan(**test_query)
            
            if "Items" in test_response and len(test_response["Items"]) > 0:
                logger.info(f"Debug: Found {len(test_response['Items'])} items with basic company_id filter")
                logger.info(f"Debug: First item keys: {list(test_response['Items'][0].keys())}")
                
                # Check data types in sample item
                sample_types = {}
                for key, value in test_response["Items"][0].items():
                    sample_types[key] = list(value.keys())[0]  # Get type (S, N, etc)
                
                return {
                    "status": "DATA_EXISTS_BUT_QUERY_TOO_RESTRICTIVE",
                    "sample_item": test_response["Items"][0],
                    "field_types": sample_types,
                    "count": len(test_response["Items"])
                }
            else:
                # Check if table exists and has items
                tables = dynamodb_client.list_tables()
                logger.info(f"Debug: Available tables: {tables['TableNames']}")
                
                if DYNAMO_TABLE_NAME in tables['TableNames']:
                    # Check table item count
                    try:
                        scan_result = dynamodb_client.scan(
                            TableName=DYNAMO_TABLE_NAME,
                            Select="COUNT"
                        )
                        logger.info(f"Debug: Total items in table: {scan_result.get('Count', 0)}")
                        
                        if scan_result.get('Count', 0) > 0:
                            # Check table schema
                            table_desc = dynamodb_client.describe_table(TableName=DYNAMO_TABLE_NAME)
                            key_schema = table_desc.get('Table', {}).get('KeySchema', [])
                            key_names = [key.get('AttributeName') for key in key_schema]
                            
                            return {
                                "status": "TABLE_HAS_DATA_BUT_NO_MATCH_FOR_COMPANY",
                                "total_items": scan_result.get('Count', 0),
                                "key_attributes": key_names
                            }
                        else:
                            return {"status": "TABLE_EXISTS_BUT_EMPTY"}
                            
                    except Exception as e:
                        logger.error(f"Debug: Error scanning table: {str(e)}")
                        return {"status": "TABLE_ACCESS_ERROR", "error": str(e)}
                else:
                    return {"status": "TABLE_DOES_NOT_EXIST"}
        
        except Exception as e:
            logger.error(f"Debug: Error in test query: {str(e)}")
            return {"status": "TEST_QUERY_FAILED", "error": str(e)}
    
    return {"status": "DATA_FOUND", "count": len(incidents)}

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "service": "incident-query-api"}

@app.post("/query")
async def query_dynamodb(request: QueryRequest):
    """API endpoint to process user query and fetch results from DynamoDB."""
    try:
        # Validate inputs
        if not request.company_id:
            raise HTTPException(status_code=400, detail="company_id is required")
        
        if not request.user_question:
            raise HTTPException(status_code=400, detail="user_question is required")
            
        # Generate the query from user input
        dynamo_query = generate_dynamodb_query(request.user_question, request.company_id)
        if not dynamo_query:
            raise HTTPException(status_code=500, detail="Failed to generate DynamoDB query")

        logger.info(f"Generated DynamoDB Query: {json.dumps(dynamo_query, indent=2)}")

        # Execute the query
        incidents = execute_dynamodb_query(dynamo_query)
        
        # Debug in logs but don't return to user
        if not incidents or len(incidents) == 0:
            debug_result = debug_dynamo_response(incidents, request.company_id)
            logger.info(f"Debug Result: {json.dumps(debug_result, indent=2)}")
            return {"message": "No incidents found."}
            
        # Format incidents for better readability
        formatted_incidents = []
        for incident in incidents:
            incident_id = incident.get("id", {}).get("S", "N/A")
            status = incident.get("status", {}).get("S", "Unknown")
            priority = incident.get("priority", {}).get("S", "Normal")
            service = incident.get("service", {}).get("S", "N/A")
            description = incident.get("description", {}).get("S", "No details available.")
            
            formatted_incident = {
                "id": incident_id,
                "status": status,
                "priority": priority,
                "service": service,
                "description": description
            }
            formatted_incidents.append(formatted_incident)

        # Return clean response with just what's needed
        return {
            "message": f"Found {len(formatted_incidents)} incident(s).",
            "incidents": formatted_incidents
        }
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional helper endpoints for debugging
@app.get("/debug/table-info")
async def get_table_info():
    """Debug endpoint to check table configuration."""
    try:
        tables = dynamodb_client.list_tables()
        
        if DYNAMO_TABLE_NAME in tables["TableNames"]:
            table_desc = dynamodb_client.describe_table(TableName=DYNAMO_TABLE_NAME)
            scan_result = dynamodb_client.scan(
                TableName=DYNAMO_TABLE_NAME,
                Select="COUNT"
            )
            
            return {
                "status": "success",
                "table_exists": True,
                "item_count": scan_result.get("Count", 0),
                "key_schema": table_desc.get("Table", {}).get("KeySchema", []),
                "region": DYNAMODB_REGION
            }
        else:
            return {
                "status": "warning",
                "table_exists": False,
                "available_tables": tables["TableNames"],
                "configured_table": DYNAMO_TABLE_NAME,
                "region": DYNAMODB_REGION
            }
    except Exception as e:
        logger.error(f"Error getting table info: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/debug/scan-sample")
async def scan_sample(company_id: str = None, limit: int = 5):
    """Debug endpoint to scan a sample of records."""
    try:
        query = {"TableName": DYNAMO_TABLE_NAME, "Limit": limit}
        
        if company_id:
            query["FilterExpression"] = "company_id = :company_id"
            query["ExpressionAttributeValues"] = {":company_id": {"S": company_id}}
            
        result = dynamodb_client.scan(**query)
        
        if result.get("Items", []):
            return {
                "status": "success",
                "count": len(result["Items"]),
                "sample": result["Items"]
            }
        else:
            return {
                "status": "warning",
                "message": "No items found with the given criteria",
                "query_used": query
            }
    except Exception as e:
        logger.error(f"Error scanning sample: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)