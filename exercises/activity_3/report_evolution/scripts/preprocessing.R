#install.packages("gapminder")
#install.packages("tidyverse")
#install.packages("countrycode")

library(gapminder)
library(tidyverse)
library(countrycode)



# We're using two datasets: gapminder and the "world" map data from ggplot. 

# Let's doublecheck our countries

countries_gap <- unique(gapminder$country)

countries_world <- unique(map_data("world")$region)

length(countries_gap) # 142
length(countries_world) #252


# Which countries occur in both datasets?

## Creating the search:

countries_search <- str_c(countries_world, collapse = "|")

## Detecting the matches and non-matches

(matches <- str_subset(countries_gap, countries_search))

(non_matches <- str_subset(countries_gap, countries_search, negate = TRUE))

# Let's explore which names aren't matching up

str_subset(countries_world, "Congo")
#[1] "Democratic Republic of the Congo"
#[2] "Republic of Congo"

str_subset(countries_world, "Ivory")
# [1] "Ivory Coast"

str_subset(countries_world, "U")
# [1] "United Arab Emirates" "UK"                  
# [3] "Uganda"               "Ukraine"             
# [5] "Uruguay"              "USA"                 
# [7] "Uzbekistan"   

str_subset(countries_world, "Korea")
#[1] "South Korea" "North Korea"

## No West Bank and Gaza on our map! Something we'll want to get more data for.

str_subset(countries_world, "Gaza")
str_subset(countries_world, "West")

# Let's do some recoding

gapminder_post <- gapminder %>%
  mutate(country = recode(country, "Congo, Dem. Rep." = "Democratic Republic of the Congo",  "Congo, Rep." = "Republic of Congo", "Cote d'Ivoire" = "Ivory Coast", "Korea, Dem. Rep." = "South Korea", "Korea, Rep." = "North Korea", "Slovak Republic" = "Slovakia", "Yemen, Rep." = "Yemen"))

world_map_data <- map_data("world") %>%
  rename(country = region) %>%
  mutate(country = recode(country, "UK" = "United Kingdom", "USA" = "United States"),
         continent = countrycode(country, origin = "country.name", destination = "continent")) 



