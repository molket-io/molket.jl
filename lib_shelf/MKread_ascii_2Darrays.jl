module MKread_ascii_2Darrays

# Read the entire file into a string array
# Taha Selim, Dec. 24th, 2024
using DelimitedFiles

#Function module file looks up the location of a module from the method table
module_file(modu) = String(first(methods(getfield(modu, :eval))).file)
# Reference: 
# https://stackoverflow.com/questions/61440940/how-to-find-the-path-of-a-package-in-julia

println("Load ascii 2Darray")

export read_ascii_with_strings, read_mixed_data

function read_ascii_with_strings(filename)
    # Read the entire file into a string array
    # Taha Selim, Dec. 24th, 2024

    data = readlines(filename)
  
    # Initialize an empty array to store the 2D array
    result = []
  
    # Iterate through the lines
    for line in data
      # Split the line by spaces
      row = split(line)
      
      # Convert the row to a vector of Any to accommodate strings and numbers
      row = convert(Vector{Any}, row)
  
      # Add the row to the result array
      push!(result, row)
    end
  
    return result
  end # function read_ascii_with_strings


function read_mixed_data(filename)
    # Read the entire file into a array of mixed data types
    # Taha Selim, Dec. 24th, 2024
    
    data_str = readlines(filename)
  
    # Initialize an empty array to store the 2D array
    data = []
  
    for line in data_str
      row_str = split(line)
      row = []
      for el in row_str
        try
          # Try parsing as Float64
          push!(row, parse(Float64, el))
        catch
          try
            # If Float64 fails, try parsing as Int
            push!(row, parse(Int, el))
          catch
            # If both fail, keep as String
            push!(row, el)
          end
        end
      end
      push!(data, row)
    end
  
    return data
  end # function read_mixed_data
  

end # module MKread_ascii_2Darrays