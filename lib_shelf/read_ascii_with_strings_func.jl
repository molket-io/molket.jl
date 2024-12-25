
## function to read 2D array of Float64 ========== ##
#using DelimitedFiles

# Example usage
#filename = "Vpot_heco2_refdata.dat"
#pot_data = readdlm(filename, ' ', Float64)

#println(data)

## ======= function to read 2D array of mixed types of float64, int64, and strings ========== ##

#using DelimitedFiles

function read_ascii_with_strings(filename)
  # Read the entire file into a string array
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
end

# Example usage
#filename = "Vpot_heco2_refdata.dat" # Replace with your actual filename
#data = read_ascii_with_strings(filename)
#println(data)


## ======= function to read 2D array of mixed types of float64 and int64 ========== ##

#using DelimitedFiles

function read_mixed_data(filename)
  # Read the entire file into a string array
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
end

# Example usage
#filename = "your_file.txt"
#data = read_mixed_data(filename)
#println(data)

