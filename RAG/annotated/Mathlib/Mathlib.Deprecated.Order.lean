@[deprecated "This was a leftover from Lean 3, and has not been needed." (since := "2024-11-26")]
instance isStrictTotalOrder_of_linearOrder [LinearOrder α] : IsStrictTotalOrder α (· < ·) where
  irrefl := lt_irrefl
  trichotomous := lt_trichotomy

