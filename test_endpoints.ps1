# PowerShell script to test Search Vector API endpoints
# Usage: .\test_endpoints.ps1

$BaseUrl = "http://127.0.0.1:5000"
$Headers = @{ "Content-Type" = "application/json" }

Write-Host "🚀 Search Vector API Endpoint Tester" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Yellow

# Function to test an endpoint
function Test-Endpoint {
    param(
        [string]$Method,
        [string]$Url,
        [string]$Body = $null,
        [string]$Description
    )
    
    try {
        Write-Host "Testing: $Description" -ForegroundColor Yellow
        
        if ($Method -eq "GET") {
            $response = Invoke-WebRequest -Uri $Url -Method $Method -ErrorAction Stop
        } else {
            $response = Invoke-WebRequest -Uri $Url -Method $Method -Body $Body -Headers $Headers -ErrorAction Stop
        }
        
        $data = $response.Content | ConvertFrom-Json
        Write-Host "✅ SUCCESS: $Description" -ForegroundColor Green
        return @{ Success = $true; Data = $data; StatusCode = $response.StatusCode }
    }
    catch {
        Write-Host "❌ FAILED: $Description - $($_.Exception.Message)" -ForegroundColor Red
        return @{ Success = $false; Data = $null; StatusCode = $null }
    }
}

# Test 1: Health Check
Write-Host "`n1. Health Check" -ForegroundColor Cyan
$healthResult = Test-Endpoint -Method "GET" -Url "$BaseUrl/health" -Description "Health endpoint"

if ($healthResult.Success) {
    Write-Host "   Status: $($healthResult.Data.status)" -ForegroundColor White
    Write-Host "   Database: $($healthResult.Data.services.database)" -ForegroundColor White
}

# Test 2: Search Endpoint
Write-Host "`n2. Search Endpoint" -ForegroundColor Cyan
$searchBody = @{
    user_id = 1
    limit = 5
} | ConvertTo-Json

$searchResult = Test-Endpoint -Method "POST" -Url "$BaseUrl/api/v1/search" -Body $searchBody -Description "Basic search"

$playerId = $null
if ($searchResult.Success -and $searchResult.Data.results.Count -gt 0) {
    $playerId = $searchResult.Data.results[0].id
    Write-Host "   Found $($searchResult.Data.total) players" -ForegroundColor White
    Write-Host "   First player: $($searchResult.Data.results[0].name) (ID: $playerId)" -ForegroundColor White
}

# Test 3: Search with Query
$querySearchBody = @{
    user_id = 1
    query = "basketball player"
    limit = 3
} | ConvertTo-Json

$queryResult = Test-Endpoint -Method "POST" -Url "$BaseUrl/api/v1/search" -Body $querySearchBody -Description "Search with query"

# Test 4: Recommendations
Write-Host "`n3. Recommendations Endpoint" -ForegroundColor Cyan
if ($playerId) {
    $recResult = Test-Endpoint -Method "GET" -Url "$BaseUrl/api/v1/recommendations/$playerId?limit=3" -Description "Get recommendations"
    
    if ($recResult.Success) {
        Write-Host "   Found $($recResult.Data.total) recommendations" -ForegroundColor White
    }
} else {
    Write-Host "⏭️  Skipped: No player ID available" -ForegroundColor Yellow
}

# Test 5: Events Endpoint
Write-Host "`n4. Events Endpoint" -ForegroundColor Cyan
if ($playerId) {
    $eventTypes = @("impression", "profile_view", "follow", "message", "save_to_playlist")
    
    foreach ($eventType in $eventTypes) {
        $eventBody = @{
            user_id = 1
            player_id = $playerId
            event_type = $eventType
            result_position = 1
        } | ConvertTo-Json
        
        $eventResult = Test-Endpoint -Method "POST" -Url "$BaseUrl/api/v1/events" -Body $eventBody -Description "Log $eventType event"
    }
} else {
    Write-Host "⏭️  Skipped: No player ID available" -ForegroundColor Yellow
}

# Test 6: Error Cases
Write-Host "`n5. Error Handling" -ForegroundColor Cyan

# Test invalid endpoint
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/v1/invalid" -Method GET -ErrorAction Stop
    Write-Host "❌ FAILED: Should have returned 404 for invalid endpoint" -ForegroundColor Red
}
catch {
    if ($_.Exception.Response.StatusCode -eq 404) {
        Write-Host "✅ SUCCESS: Correctly returned 404 for invalid endpoint" -ForegroundColor Green
    } else {
        Write-Host "❌ FAILED: Unexpected error for invalid endpoint" -ForegroundColor Red
    }
}

# Test invalid event type
$invalidEventBody = @{
    user_id = 1
    player_id = "test_id"
    event_type = "invalid_event"
    result_position = 1
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/v1/events" -Method POST -Body $invalidEventBody -Headers $Headers -ErrorAction Stop
    Write-Host "❌ FAILED: Should have rejected invalid event type" -ForegroundColor Red
}
catch {
    if ($_.Exception.Response.StatusCode -eq 400) {
        Write-Host "✅ SUCCESS: Correctly rejected invalid event type" -ForegroundColor Green
    } else {
        Write-Host "❌ FAILED: Unexpected error for invalid event type" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Testing Complete!" -ForegroundColor Green
Write-Host "Check the output above for any failed tests." -ForegroundColor White