from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from .. import db
from ..auth import create_token, current_user, hash_password, verify_password
from ..schemas import LoginIn, RegisterIn, TokenOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn):
    try:
        with db.session() as con:
            cur = con.execute("INSERT INTO users(email, name, password_hash) VALUES (?,?,?)",
                              (body.email, body.name.strip(), hash_password(body.password)))
            uid = cur.lastrowid
    except sqlite3.IntegrityError as e:
        raise HTTPException(409, "An account with this email already exists") from e
    return {"access_token": create_token(uid), "user": {"id": uid, "email": body.email, "name": body.name}}


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn):
    with db.session() as con:
        u = db.row(con.execute("SELECT * FROM users WHERE email=?", (body.email.strip().lower(),)).fetchone())
    if not u or not verify_password(body.password, u["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    return {"access_token": create_token(u["id"]), "user": {"id": u["id"], "email": u["email"], "name": u["name"]}}


@router.get("/me")
def me(user=Depends(current_user)):
    return user
