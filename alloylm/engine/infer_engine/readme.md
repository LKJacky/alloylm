# Framework Overview:

| request session
|
api_server
|
| schedule session
↓
scheduler
|
| engine session
↓
engine




Api server
    - messages
    - prompt tokens

Scheduler
    - forwarded tokens
    - prompt tokens

Engine
    - forwarded tokens
    - prompt tokens