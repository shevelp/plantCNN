library(shiny)
library(keras)

# Load the model
model <- load_model_tf("/home/sergio/Projects/Agro/postcovid19/model/cnn-agro/")
classes <- read.csv("./classes.csv")


# Define the UI
ui <- fluidPage(
  # App title ----
  titlePanel("Plant CNN!"),
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    # Sidebar panel for inputs ----
    sidebarPanel(
      # Input: File upload
      fileInput("image_path", label = "Input a JPEG image"),
      plotOutput(outputId = "image")
    ),
    # Main panel for displaying outputs ----
    mainPanel(
      # Output: Histogram ----
      dataTableOutput(outputId = "prediction")
     
    )
  )
)


# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  image <- reactive({
    req(input$image_path)
    jpeg::readJPEG(input$image_path$datapath)
  })
  
  output$prediction <- renderDataTable({
    
    img <- image() %>% 
      EBImage::resize(w = 20,h = 20) %>%
      array_reshape(., dim = c(1,20,20,3))
    
    img <- img * 1/255
    
    preds <- model %>% predict_proba(test_image) %>%
      as.data.frame() %>%
      t() %>%
      as.data.frame() %>%
      format(.,scientific = F)
    
    rownames(preds) <- classes$classes
    names(classes) <- "score"
    print.table(preds)
  })
  
  output$image <- renderPlot({
    plot(as.raster(image()))
  })
  
}

shinyApp(ui, server)
