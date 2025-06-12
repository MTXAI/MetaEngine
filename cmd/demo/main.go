package main

import (
	"github.com/gin-gonic/gin"
	_ "github.com/lestrrat-go/file-rotatelogs"
)

type Server struct {
	*gin.Engine
}
