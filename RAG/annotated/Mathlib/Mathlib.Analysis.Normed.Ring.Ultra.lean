lemma norm_add_one_le_max_norm_one (x : R) :
    ‖x + 1‖ ≤ max ‖x‖ 1 := by
  /-
    R : Type u_1
    inst✝² : SeminormedRing R
    inst✝¹ : NormOneClass R
    inst✝ : IsUltrametricDist R
    x : R
    ⊢ LE.le (Norm.norm (HAdd.hAdd x 1)) (Max.max (Norm.norm x) 1)
  -/
  simpa only [le_max_iff, norm_one] using norm_add_le_max x 1
  /-
    🎉 no goals
  -/


lemma nnnorm_add_one_le_max_nnnorm_one (x : R) :
    ‖x + 1‖₊ ≤ max ‖x‖₊ 1 :=
  norm_add_one_le_max_norm_one _


lemma nnnorm_natCast_le_one (n : ℕ) :
    ‖(n : R)‖₊ ≤ 1 := by
  induction n with
  | zero => simp only [Nat.cast_zero, nnnorm_zero, zero_le]
  | succ n hn => simpa only [Nat.cast_add, Nat.cast_one, hn, max_eq_right] using
    nnnorm_add_one_le_max_nnnorm_one (n : R)


lemma norm_natCast_le_one (n : ℕ) :
    ‖(n : R)‖ ≤ 1 :=
  nnnorm_natCast_le_one R n


lemma nnnorm_intCast_le_one (z : ℤ) :
    ‖(z : R)‖₊ ≤ 1 := by
  /-
    R : Type u_1
    inst✝² : SeminormedRing R
    inst✝¹ : NormOneClass R
    inst✝ : IsUltrametricDist R
    z : Int
    ⊢ LE.le (NNNorm.nnnorm ↑z) 1
  -/
  induction z <;>
  simpa only [Int.ofNat_eq_coe, Int.cast_natCast, Int.cast_negSucc, Nat.cast_one, nnnorm_neg]
    using nnnorm_natCast_le_one _ _


lemma norm_intCast_le_one (z : ℤ) :
    ‖(z : R)‖ ≤ 1 :=
  nnnorm_intCast_le_one _ z


