/-- A function that is inverse to `f` near `a`. -/
abbrev localInverse : 𝕜 → 𝕜 :=
  (hf.hasStrictFDerivAt_equiv hf').localInverse _ _ _


theorem map_nhds_eq : map f (𝓝 a) = 𝓝 (f a) :=
  (hf.hasStrictFDerivAt_equiv hf').map_nhds_eq_of_equiv


theorem to_localInverse : HasStrictDerivAt (hf.localInverse f f' a hf') f'⁻¹ (f a) :=
  (hf.hasStrictFDerivAt_equiv hf').to_localInverse


theorem to_local_left_inverse {g : 𝕜 → 𝕜} (hg : ∀ᶠ x in 𝓝 a, g (f x) = x) :
    HasStrictDerivAt g f'⁻¹ (f a) :=
  (hf.hasStrictFDerivAt_equiv hf').to_local_left_inverse hg


/-- If a function has a non-zero strict derivative at all points, then it is an open map. -/
theorem isOpenMap_of_hasStrictDerivAt {f' : 𝕜 → 𝕜}
    (hf : ∀ x, HasStrictDerivAt f (f' x) x) (h0 : ∀ x, f' x ≠ 0) : IsOpenMap f :=
  isOpenMap_iff_nhds_le.2 fun x => ((hf x).map_nhds_eq (h0 x)).ge

@[deprecated (since := "2024-03-23")]
alias open_map_of_strict_deriv := isOpenMap_of_hasStrictDerivAt

