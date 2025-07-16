import os
import argparse
from app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DiseaseX Healthcare API server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, 
                        default=int(os.environ.get('PORT', 5000)),
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Check if we're in production mode
    is_production = os.environ.get('FLASK_ENV') == 'production'
    debug_mode = args.debug and not is_production
    
    print(f"\n\n=== Starting DiseaseX Healthcare API server on {args.host}:{args.port} ===\n")
    print(f"Environment: {'Production' if is_production else 'Development'}")
    print(f"Debug mode: {'Off' if is_production else 'On'}\n")
    
    app.run(host=args.host, port=args.port, debug=debug_mode)
