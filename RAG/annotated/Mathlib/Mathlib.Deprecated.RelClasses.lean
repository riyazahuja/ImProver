@[deprecated "No deprecation message was provided." (since := "2024-07-30")]
theorem IsTotalPreorder.swap (r) [IsTotalPreorder α r] : IsTotalPreorder α (swap r) :=
  { @IsPreorder.swap α r _, @IsTotal.swap α r _ with }


@[deprecated "No deprecation message was provided." (since := "2024-08-22")]
instance [LinearOrder α] : IsTotalPreorder α (· ≤ ·) where

@[deprecated "No deprecation message was provided." (since := "2024-08-22")]
instance [LinearOrder α] : IsTotalPreorder α (· ≥ ·) where


@[deprecated "No deprecation message was provided." (since := "2024-07-30")]
                                                         /-
                                                           α : Type u
                                                           inst✝ : LinearOrder α
                                                           ⊢ IsIncompTrans α fun x1 x2 => LT.lt x1 x2
                                                         -/
instance [LinearOrder α] : IsIncompTrans α (· < ·) := by infer_instance
                                                         /-
                                                           🎉 no goals
                                                         -/

