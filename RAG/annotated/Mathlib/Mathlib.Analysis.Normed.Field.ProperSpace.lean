/-- A weakly locally compact normed field is proper.
This is a specialization of `ProperSpace.of_locallyCompactSpace`
which holds for `NormedSpace`s but requires more imports. -/
lemma ProperSpace.of_nontriviallyNormedField_of_weaklyLocallyCompactSpace
    (𝕜 : Type*) [NontriviallyNormedField 𝕜] [WeaklyLocallyCompactSpace 𝕜] :
    ProperSpace 𝕜 := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : WeaklyLocallyCompactSpace 𝕜
    ⊢ ProperSpace 𝕜
  -/
  rcases exists_isCompact_closedBall (0 : 𝕜) with ⟨r, rpos, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : WeaklyLocallyCompactSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    hr : IsCompact (Metric.closedBall 0 r)
    ⊢ ProperSpace 𝕜
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  have hC n : IsCompact (closedBall (0 : 𝕜) (‖c‖^n * r)) := by
    have : c ^ n ≠ 0 := pow_ne_zero _ <| fun h ↦ by simp [h, zero_le_one.not_lt] at hc
    convert hr.smul (c ^ n)
    ext
    simp only [mem_closedBall, dist_zero_right, Set.mem_smul_set_iff_inv_smul_mem₀ this,
      smul_eq_mul, norm_mul, norm_inv, norm_pow,
      inv_mul_le_iff₀ (by simpa only [norm_pow] using norm_pos_iff.mpr this)]
  have hTop : Tendsto (fun n ↦ ‖c‖^n * r) atTop atTop :=
    Tendsto.atTop_mul_const rpos (tendsto_pow_atTop_atTop_of_one_lt hc)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : WeaklyLocallyCompactSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    hr : IsCompact (Metric.closedBall 0 r)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hC : ∀ (n : Nat), IsCompact (Metric.closedBall 0 (HMul.hMul (HPow.hPow (Norm.n …
    hTop : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (Norm.norm c) n) r) Filte …
    ⊢ ProperSpace 𝕜
  -/
  exact .of_seq_closedBall hTop (Eventually.of_forall hC)
  /-
    🎉 no goals
  -/

