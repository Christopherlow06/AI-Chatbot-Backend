from jose import JWTError, jwt
from datetime import timedelta, datetime , timezone
from fastapi import HTTPException, Request, status, Depends
from fastapi.security import OAuth2PasswordBearer
import os

SECRET_KEY = os.environ.get("SECRET_KEY", "c54593b4ac10ed601d5eb1626ea98e4a1169fed9be6a5d2607c4308d3e057bb43254f1297348dfee5e3ad8d3fab3a5cb6cac9fdf112fac529cddbefa7befa5056397a5502a11e7771e5da692bad7ea3f2d73d867241219e354ab5fbc01087a35ca6519b069c4bb44c6a5941e48a312fcbb3250c0079159bbcc4d43d07c4521ce") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 240
os.environ['VERIFY_SSL_CERTS'] = 'False'

# Reuse the oauth2_scheme defined above
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt=jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        login_id: str = payload.get("sub")
        if login_id is None:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception    

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = verify_token(token, credentials_exception)
    return payload.get("sub")  # login_id

async def get_current_role(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = verify_token(token, credentials_exception)
    return payload.get("role")