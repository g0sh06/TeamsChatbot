# Try to create a simple storage account in different regions to test
TEST_REGIONS=("eastus" "westus" "centralus" "northcentralus" "southcentralus" "westus2" "westus3")

for region in "${TEST_REGIONS[@]}"; do
    echo "Testing region: $region"
    storage_name="teststorage$(date +%s | tail -c 4)"
    az storage account create --name "$storage_name" --resource-group ChatbotRG --sku Standard_LRS --location "$region" --only-show-errors
    if [ $? -eq 0 ]; then
        echo "✅ Region $region is allowed!"
        ALLOWED_REGION="$region"
        # Clean up the test storage
        az storage account delete --name "$storage_name" --resource-group ChatbotRG --yes
        break
    else
        echo "❌ Region $region is not allowed"
    fi
done

if [ -z "$ALLOWED_REGION" ]; then
    echo "Trying EU regions..."
    EU_REGIONS=("northeurope" "westeurope" "francecentral" "germanywestcentral" "uksouth")
    for region in "${EU_REGIONS[@]}"; do
        echo "Testing region: $region"
        storage_name="teststorage$(date +%s | tail -c 4)"
        az storage account create --name "$storage_name" --resource-group ChatbotRG --sku Standard_LRS --location "$region" --only-show-errors
        if [ $? -eq 0 ]; then
            echo "✅ Region $region is allowed!"
            ALLOWED_REGION="$region"
            az storage account delete --name "$storage_name" --resource-group ChatbotRG --yes
            break
        else
            echo "❌ Region $region is not allowed"
        fi
    done
fi

if [ -z "$ALLOWED_REGION" ]; then
    echo "Could not find an allowed region. Using ngrok approach instead."
    # We'll use the ngrok approach below
fi