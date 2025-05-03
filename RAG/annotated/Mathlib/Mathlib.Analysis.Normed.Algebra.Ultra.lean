variable (L) in
/--
The other direction of `IsUltrametricDist.of_normedAlgebra`.
Let `K` be a normed field. If a seminormed ring `L` is a normed `K`-algebra, and `‖1‖ = 1` in `L`,
then `K` is ultrametric (i.e. the norm on `L` is nonarchimedean) if `F` is.
This can be further generalized to the case where `‖1‖ ≠ 0` in `L`.
-/
theorem IsUltrametricDist.of_normedAlgebra' [SeminormedRing L] [NormOneClass L] [NormedAlgebra K L]
    [h : IsUltrametricDist L] : IsUltrametricDist K :=
  ⟨fun x y z => by
    /-
      K : Type u_1
      L : Type u_2
      inst✝³ : NormedField K
      inst✝² : SeminormedRing L
      inst✝¹ : NormOneClass L
      inst✝ : NormedAlgebra K L
      h : IsUltrametricDist L
      x y z : K
      ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x y) (Dist.dist y z))
    -/
    simpa using h.dist_triangle_max (algebraMap K L x) (algebraMap K L y) (algebraMap K L z)⟩
    /-
      🎉 no goals
    -/


variable (K) in
/--
Let `K` be a normed field. If a normed division ring `L` is a normed `K`-algebra,
then `L` is ultrametric (i.e. the norm on `L` is nonarchimedean) if `K` is.
-/
theorem IsUltrametricDist.of_normedAlgebra [NormedDivisionRing L] [NormedAlgebra K L]
    [h : IsUltrametricDist K] : IsUltrametricDist L := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedDivisionRing L
    inst✝ : NormedAlgebra K L
    h : IsUltrametricDist K
    ⊢ IsUltrametricDist L
  -/
  rw [isUltrametricDist_iff_forall_norm_natCast_le_one] at h ⊢
  /-
    K : Type u_1
    L : Type u_2
    inst✝² : NormedField K
    inst✝¹ : NormedDivisionRing L
    inst✝ : NormedAlgebra K L
    h : ∀ (n : Nat), LE.le (Norm.norm ↑n) 1
    ⊢ ∀ (n : Nat), LE.le (Norm.norm ↑n) 1
  -/
  exact fun n => (algebraMap.coe_natCast (R := K) (A := L) n) ▸ norm_algebraMap' L (n : K) ▸ h n
  /-
    🎉 no goals
  -/


variable (K L) in
/--
Let `K` be a normed field. If a normed division ring `L` is a normed `K`-algebra,
then `L` is ultrametric (i.e. the norm on `L` is nonarchimedean) if and only if `K` is.
-/
theorem IsUltrametricDist.normedAlgebra_iff [NormedDivisionRing L] [NormedAlgebra K L] :
    IsUltrametricDist L ↔ IsUltrametricDist K :=
  ⟨fun _ => IsUltrametricDist.of_normedAlgebra' L, fun _ => IsUltrametricDist.of_normedAlgebra K⟩

