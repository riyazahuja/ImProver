private lemma chainLength_aux (hα : α.IsNonZero) {x} (hx : x ∈ rootSpace H (chainTop α β)) :
    ∃ n : ℕ, n • x = ⁅coroot α, x⁆ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
  -/
  by_cases hx' : x = 0
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      x : L
      hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
      hx' : Eq x 0
      ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
    -/
  · exact ⟨0, by simp [hx']⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    hx' : Not (Eq x 0)
    ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
  -/
  obtain ⟨h, e, f, isSl2, he, hf⟩ := exists_isSl2Triple_of_weight_isNonZero hα
  /-
    case neg.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    hx' : Not (Eq x 0)
    h e f : L
    isSl2 : IsSl2Triple h e f
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
  -/
  obtain rfl := isSl2.h_eq_coroot hα he hf
  have : isSl2.HasPrimitiveVectorWith x (chainTop α β (coroot α)) :=
    have := lie_mem_genWeightSpace_of_mem_genWeightSpace he hx
    ⟨hx', by rw [← lie_eq_smul_of_mem_rootSpace hx]; rfl,
      by rwa [genWeightSpace_add_chainTop α β hα] at this⟩
  /-
    case neg.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    hx' : Not (Eq x 0)
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    this : isSl2.HasPrimitiveVectorWith x ((LieModule.chainTop (⇑α) β) (LieAlgebra …
    ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
  -/
  obtain ⟨μ, hμ⟩ := this.exists_nat
  /-
    case neg.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    hx' : Not (Eq x 0)
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    this : isSl2.HasPrimitiveVectorWith x ((LieModule.chainTop (⇑α) β) (LieAlgebra …
    μ : Nat
    hμ : Eq ((LieModule.chainTop (⇑α) β) (LieAlgebra.IsKilling.coroot α)) ↑μ
    ⊢ Exists fun n => Eq (HSMul.hSMul n x) (Bracket.bracket (LieAlgebra.IsKilling. …
  -/
  exact ⟨μ, by rw [← Nat.cast_smul_eq_nsmul K, ← hμ, lie_eq_smul_of_mem_rootSpace hx]⟩
  /-
    🎉 no goals
  -/


/-- The length of the `α`-chain through `β`. See `chainBotCoeff_add_chainTopCoeff`. -/
def chainLength (α β : Weight K H L) : ℕ :=
  letI := Classical.propDecidable
  if hα : α.IsZero then 0 else
    (chainLength_aux α β hα (chainTop α β).exists_ne_zero.choose_spec.1).choose


lemma chainLength_of_isZero (hα : α.IsZero) : chainLength α β = 0 := dif_pos hα


lemma chainLength_nsmul {x} (hx : x ∈ rootSpace H (chainTop α β)) :
    chainLength α β • x = ⁅coroot α, x⁆ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    ⊢ Eq (HSMul.hSMul (LieAlgebra.IsKilling.chainLength α β) x) (Bracket.bracket ( …
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
      hα : α.IsZero
      ⊢ Eq (HSMul.hSMul (LieAlgebra.IsKilling.chainLength α β) x) (Bracket.bracket ( …
    -/
  · rw [coroot_eq_zero_iff.mpr hα, chainLength_of_isZero _ _ hα, zero_smul, zero_lie]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    hα : Not α.IsZero
    ⊢ Eq (HSMul.hSMul (LieAlgebra.IsKilling.chainLength α β) x) (Bracket.bracket ( …
  -/
  let x' := (chainTop α β).exists_ne_zero.choose
  have h : x' ∈ rootSpace H (chainTop α β) ∧ x' ≠ 0 :=
    (chainTop α β).exists_ne_zero.choose_spec
  obtain ⟨k, rfl⟩ : ∃ k : K, k • x' = x := by
    simpa using (finrank_eq_one_iff_of_nonzero' ⟨x', h.1⟩ (by simpa using h.2)).mp
      (finrank_rootSpace_eq_one _ (chainTop_isNonZero α β hα)) ⟨_, hx⟩
  /-
    case neg.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    x' : L := ⋯.choose
    h : And (Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β))  …
    k : K
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) (HSM …
    ⊢ Eq (HSMul.hSMul (LieAlgebra.IsKilling.chainLength α β) (HSMul.hSMul k x')) ( …
  -/
  rw [lie_smul, smul_comm, chainLength, dif_neg hα, (chainLength_aux α β hα h.1).choose_spec]
  /-
    🎉 no goals
  -/


lemma chainLength_smul {x} (hx : x ∈ rootSpace H (chainTop α β)) :
    (chainLength α β : K) • x = ⁅coroot α, x⁆ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) x
    ⊢ Eq (HSMul.hSMul (↑(LieAlgebra.IsKilling.chainLength α β)) x) (Bracket.bracke …
  -/
  rw [Nat.cast_smul_eq_nsmul, chainLength_nsmul _ _ hx]
  /-
    🎉 no goals
  -/


lemma apply_coroot_eq_cast' :
    β (coroot α) = ↑(chainLength α β - 2 * chainTopCoeff α β : ℤ) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (β (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.c …
  -/
  by_cases hα : α.IsZero
  · rw [coroot_eq_zero_iff.mpr hα, chainLength, dif_pos hα, hα.eq, chainTopCoeff_zero, map_zero,
      CharP.cast_eq_zero, mul_zero, sub_self, Int.cast_zero]
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    ⊢ Eq (β (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.c …
  -/
  obtain ⟨x, hx, x_ne0⟩ := (chainTop α β).exists_ne_zero
  /-
    case neg.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) β)) x
    x_ne0 : Ne x 0
    ⊢ Eq (β (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.c …
  -/
  have := chainLength_smul _ _ hx
  rw [lie_eq_smul_of_mem_rootSpace hx, ← sub_eq_zero, ← sub_smul,
    smul_eq_zero_iff_left x_ne0, sub_eq_zero, coe_chainTop', nsmul_eq_mul, Pi.natCast_def,
    Pi.add_apply, Pi.mul_apply, root_apply_coroot hα] at this
  simp only [Int.cast_sub, Int.cast_natCast, Int.cast_mul, Int.cast_ofNat, eq_sub_iff_add_eq',
    this, mul_comm (2 : K)]


lemma rootSpace_neg_nsmul_add_chainTop_of_le {n : ℕ} (hn : n ≤ chainLength α β) :
    rootSpace H (- (n • α) + chainTop α β) ≠ ⊥ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Nat
      hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
      hα : α.IsZero
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
    -/
  · simpa only [hα.eq, smul_zero, neg_zero, chainTop_zero, zero_add, ne_eq] using β.2
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    hα : Not α.IsZero
    ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  obtain ⟨x, hx, x_ne0⟩ := (chainTop α β).exists_ne_zero
  /-
    case neg.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    hα : Not α.IsZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) β)) x
    x_ne0 : Ne x 0
    ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  obtain ⟨h, e, f, isSl2, he, hf⟩ := exists_isSl2Triple_of_weight_isNonZero hα
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    hα : Not α.IsZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) β)) x
    x_ne0 : Ne x 0
    h e f : L
    isSl2 : IsSl2Triple h e f
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  obtain rfl := isSl2.h_eq_coroot hα he hf
  have prim : isSl2.HasPrimitiveVectorWith x (chainLength α β : K) :=
    have := lie_mem_genWeightSpace_of_mem_genWeightSpace he hx
    ⟨x_ne0, (chainLength_smul _ _ hx).symm, by rwa [genWeightSpace_add_chainTop _ _ hα] at this⟩
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    hα : Not α.IsZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) β)) x
    x_ne0 : Ne x 0
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α β)
    ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  simp only [← smul_neg, ne_eq, LieSubmodule.eq_bot_iff, not_forall]
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Nat
    hn : LE.le n (LieAlgebra.IsKilling.chainLength α β)
    hα : Not α.IsZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) β)) x
    x_ne0 : Ne x 0
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α β)
    ⊢ Exists fun x => Exists fun x_1 => Not (Eq x 0)
  -/
  exact ⟨_, toEnd_pow_apply_mem hf hx n, prim.pow_toEnd_f_ne_zero_of_eq_nat rfl hn⟩
  /-
    🎉 no goals
  -/


lemma rootSpace_neg_nsmul_add_chainTop_of_lt (hα : α.IsNonZero) {n : ℕ} (hn : chainLength α β < n) :
    rootSpace H (- (n • α) + chainTop α β) = ⊥ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    ⊢ Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModu …
  -/
  by_contra e
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    e : Not (Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑( …
    ⊢ False
  -/
  let W : Weight K H L := ⟨_, e⟩
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    e : Not (Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑( …
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    ⊢ False
  -/
  have hW : (W : H → K) = - (n • α) + chainTop α β := rfl
  have H₁ : 1 + n + chainTopCoeff (-α) W ≤ chainLength (-α) W := by
    have := apply_coroot_eq_cast' (-α) W
    simp only [coroot_neg, map_neg, hW, nsmul_eq_mul, Pi.natCast_def, coe_chainTop, zsmul_eq_mul,
      Int.cast_natCast, Pi.add_apply, Pi.neg_apply, Pi.mul_apply, root_apply_coroot hα, mul_two,
      neg_add_rev, apply_coroot_eq_cast' α β, Int.cast_sub, Int.cast_mul, Int.cast_ofNat,
      mul_comm (2 : K), add_sub_cancel, neg_neg, add_sub, Nat.cast_inj,
      eq_sub_iff_add_eq, ← Nat.cast_add, ← sub_eq_neg_add, sub_eq_iff_eq_add] at this
    omega
  have H₂ : ((1 + n + chainTopCoeff (-α) W) • α + chainTop (-α) W : H → K) =
      (chainTopCoeff α β + 1) • α + β := by
    simp only [Weight.coe_neg, ← Nat.cast_smul_eq_nsmul ℤ, Nat.cast_add, Nat.cast_one, coe_chainTop,
      smul_neg, ← neg_smul, hW, ← add_assoc, ← add_smul, ← sub_eq_add_neg]
    congr 2
    ring
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    e : Not (Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑( …
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    hW : Eq (⇑W) (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModule.chainTop (⇑α) …
    H₁ : LE.le (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTopCoeff (⇑(Neg.neg α))  …
    H₂ : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTop …
    ⊢ False
  -/
  have := rootSpace_neg_nsmul_add_chainTop_of_le (-α) W H₁
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    e : Not (Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑( …
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    hW : Eq (⇑W) (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModule.chainTop (⇑α) …
    H₁ : LE.le (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTopCoeff (⇑(Neg.neg α))  …
    H₂ : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTop …
    this : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul (HAdd.hAdd  …
    ⊢ False
  -/
  rw [Weight.coe_neg, ← smul_neg, neg_neg, ← Weight.coe_neg, H₂] at this
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) n
    e : Not (Eq (LieAlgebra.rootSpace H (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑( …
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    hW : Eq (⇑W) (HAdd.hAdd (Neg.neg (HSMul.hSMul n ⇑α)) ⇑(LieModule.chainTop (⇑α) …
    H₁ : LE.le (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTopCoeff (⇑(Neg.neg α))  …
    H₂ : Eq (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (HAdd.hAdd 1 n) (LieModule.chainTop …
    this : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (LieModul …
    ⊢ False
  -/
  exact this (genWeightSpace_chainTopCoeff_add_one_nsmul_add α β hα)
  /-
    🎉 no goals
  -/


lemma chainTopCoeff_le_chainLength : chainTopCoeff α β ≤ chainLength α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ LE.le (LieModule.chainTopCoeff (⇑α) β) (LieAlgebra.IsKilling.chainLength α β)
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsZero
      ⊢ LE.le (LieModule.chainTopCoeff (⇑α) β) (LieAlgebra.IsKilling.chainLength α β)
    -/
  · simp only [hα.eq, chainTopCoeff_zero, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    ⊢ LE.le (LieModule.chainTopCoeff (⇑α) β) (LieAlgebra.IsKilling.chainLength α β)
  -/
  rw [← not_lt, ← Nat.succ_le]
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    ⊢ Not (LE.le (LieAlgebra.IsKilling.chainLength α β).succ (LieModule.chainTopCo …
  -/
  intro e
  apply genWeightSpace_nsmul_add_ne_bot_of_le α β
    (Nat.sub_le (chainTopCoeff α β) (chainLength α β).succ)
  rw [← Nat.cast_smul_eq_nsmul ℤ, Nat.cast_sub e, sub_smul, sub_eq_neg_add,
    add_assoc, ← coe_chainTop, Nat.cast_smul_eq_nsmul]
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    e : LE.le (LieAlgebra.IsKilling.chainLength α β).succ (LieModule.chainTopCoeff …
    ⊢ Eq (LieModule.genWeightSpace L (HAdd.hAdd (Neg.neg (HSMul.hSMul (LieAlgebra. …
  -/
  exact rootSpace_neg_nsmul_add_chainTop_of_lt α β hα (Nat.lt_succ_self _)
  /-
    🎉 no goals
  -/


lemma chainBotCoeff_add_chainTopCoeff :
    chainBotCoeff α β + chainTopCoeff α β = chainLength α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCoeff (⇑α) …
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsZero
      ⊢ Eq (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCoeff (⇑α) …
    -/
  · rw [hα.eq, chainTopCoeff_zero, chainBotCoeff_zero, zero_add, chainLength_of_isZero α β hα]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Not α.IsZero
    ⊢ Eq (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCoeff (⇑α) …
  -/
  apply le_antisymm
  · rw [← Nat.le_sub_iff_add_le (chainTopCoeff_le_chainLength α β),
      ← not_lt, ← Nat.succ_le, chainBotCoeff, ← Weight.coe_neg]
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      ⊢ Not (LE.le (HSub.hSub (LieAlgebra.IsKilling.chainLength α β) (LieModule.chai …
    -/
    intro e
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      e : LE.le (HSub.hSub (LieAlgebra.IsKilling.chainLength α β) (LieModule.chainTo …
      ⊢ False
    -/
    apply genWeightSpace_nsmul_add_ne_bot_of_le _ _ e
    rw [← Nat.cast_smul_eq_nsmul ℤ, Nat.cast_succ, Nat.cast_sub (chainTopCoeff_le_chainLength α β),
      LieModule.Weight.coe_neg, smul_neg, ← neg_smul, neg_add_rev, neg_sub, sub_eq_neg_add,
      ← add_assoc, ← neg_add_rev, add_smul, add_assoc, ← coe_chainTop, neg_smul,
      ← @Nat.cast_one ℤ, ← Nat.cast_add, Nat.cast_smul_eq_nsmul]
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      e : LE.le (HSub.hSub (LieAlgebra.IsKilling.chainLength α β) (LieModule.chainTo …
      ⊢ Eq (LieModule.genWeightSpace L (HAdd.hAdd (Neg.neg (HSMul.hSMul (HAdd.hAdd ( …
    -/
    exact rootSpace_neg_nsmul_add_chainTop_of_lt α β hα (Nat.lt_succ_self _)
    /-
      🎉 no goals
    -/
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      ⊢ LE.le (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (LieModule.chainBotC …
    -/
  · rw [← not_lt]
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      ⊢ Not (LT.lt (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCo …
    -/
    intro e
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      e : LT.lt (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCoeff …
      ⊢ False
    -/
    apply rootSpace_neg_nsmul_add_chainTop_of_le α β e
    rw [← Nat.succ_add, ← Nat.cast_smul_eq_nsmul ℤ, ← neg_smul, coe_chainTop, ← add_assoc,
      ← add_smul, Nat.cast_add, neg_add, add_assoc, neg_add_cancel, add_zero, neg_smul, ← smul_neg,
      Nat.cast_smul_eq_nsmul]
    /-
      case neg.a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      e : LT.lt (HAdd.hAdd (LieModule.chainBotCoeff (⇑α) β) (LieModule.chainTopCoeff …
      ⊢ Eq (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (LieModule.chainBotCoeff  …
    -/
    exact genWeightSpace_chainTopCoeff_add_one_nsmul_add (-α) β (Weight.IsNonZero.neg hα)
    /-
      🎉 no goals
    -/


lemma chainTopCoeff_add_chainBotCoeff :
    chainTopCoeff α β + chainBotCoeff α β = chainLength α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (HAdd.hAdd (LieModule.chainTopCoeff (⇑α) β) (LieModule.chainBotCoeff (⇑α) …
  -/
  rw [add_comm, chainBotCoeff_add_chainTopCoeff]
  /-
    🎉 no goals
  -/


lemma chainBotCoeff_le_chainLength : chainBotCoeff α β ≤ chainLength α β :=
  (Nat.le_add_left _ _).trans_eq (chainTopCoeff_add_chainBotCoeff α β)


@[simp]
lemma chainLength_neg :
    chainLength (-α) β = chainLength α β := by
  rw [← chainBotCoeff_add_chainTopCoeff, ← chainBotCoeff_add_chainTopCoeff, add_comm,
    Weight.coe_neg, chainTopCoeff_neg, chainBotCoeff_neg]


@[simp]
lemma chainLength_zero [Nontrivial L] : chainLength 0 β = 0 := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    ⊢ Eq (LieAlgebra.IsKilling.chainLength 0 β) 0
  -/
  simp [← chainBotCoeff_add_chainTopCoeff]
  /-
    🎉 no goals
  -/


/-- If `β - qα ... β ... β + rα` is the `α`-chain through `β`, then
  `β (coroot α) = q - r`. In particular, it is an integer. -/
lemma apply_coroot_eq_cast :
    β (coroot α) = (chainBotCoeff α β - chainTopCoeff α β : ℤ) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (β (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub ↑(LieModule.chainBotCoeff …
  -/
  rw [apply_coroot_eq_cast', ← chainTopCoeff_add_chainBotCoeff]; congr 1; omega
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma le_chainBotCoeff_of_rootSpace_ne_top
    (hα : α.IsNonZero) (n : ℤ) (hn : rootSpace H (-n • α + β) ≠ ⊥) :
    n ≤ chainBotCoeff α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Int
    hn : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (Neg.neg n) ⇑α) ⇑β)) B …
    ⊢ LE.le n ↑(LieModule.chainBotCoeff (⇑α) β)
  -/
  contrapose! hn
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Int
    hn : LT.lt (↑(LieModule.chainBotCoeff (⇑α) β)) n
    ⊢ Eq (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (Neg.neg n) ⇑α) ⇑β)) Bot. …
  -/
  lift n to ℕ using (Nat.cast_nonneg _).trans hn.le
  rw [Nat.cast_lt, ← @Nat.add_lt_add_iff_right (chainTopCoeff α β),
    chainBotCoeff_add_chainTopCoeff] at hn
  /-
    case intro
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Nat
    hn : LT.lt (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n (LieModule.chai …
    ⊢ Eq (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (Neg.neg ↑n) ⇑α) ⇑β)) Bot …
  -/
  have := rootSpace_neg_nsmul_add_chainTop_of_lt α β hα hn
  rwa [← Nat.cast_smul_eq_nsmul ℤ, ← neg_smul, coe_chainTop, ← add_assoc,
    ← add_smul, Nat.cast_add, neg_add, add_assoc, neg_add_cancel, add_zero] at this


/-- Members of the `α`-chain through `β` are the only roots of the form `β - kα`. -/
lemma rootSpace_zsmul_add_ne_bot_iff (hα : α.IsNonZero) (n : ℤ) :
    rootSpace H (n • α + β) ≠ ⊥ ↔ n ≤ chainTopCoeff α β ∧ -n ≤ chainBotCoeff α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Int
    ⊢ Iff (Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot)  …
  -/
  constructor
    /-
      case mp
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot → And  …
    -/
  · refine (fun hn ↦ ⟨?_, le_chainBotCoeff_of_rootSpace_ne_top α β hα _ (by rwa [neg_neg])⟩)
    /-
      case mp
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      hn : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
      ⊢ LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)
    -/
    rw [← chainBotCoeff_neg, ← Weight.coe_neg]
    /-
      case mp
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      hn : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
      ⊢ LE.le n ↑(LieModule.chainBotCoeff (⇑(Neg.neg α)) β)
    -/
    apply le_chainBotCoeff_of_rootSpace_ne_top _ _ hα.neg
    /-
      case mp.hn
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      hn : Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (Neg.neg n) ⇑(Neg.neg α)) …
    -/
    rwa [neg_smul, Weight.coe_neg, smul_neg, neg_neg]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      ⊢ And (LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)) (LE.le (Neg.neg n) ↑(LieModu …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      h₁ : LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)
      h₂ : LE.le (Neg.neg n) ↑(LieModule.chainBotCoeff (⇑α) β)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
    -/
    set k := chainTopCoeff α β - n with hk; clear_value k
    /-
      case mpr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      h₁ : LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)
      h₂ : LE.le (Neg.neg n) ↑(LieModule.chainBotCoeff (⇑α) β)
      k : Int
      hk : Eq k (HSub.hSub (↑(LieModule.chainTopCoeff (⇑α) β)) n)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
    -/
    lift k to ℕ using (by rw [hk, le_sub_iff_add_le, zero_add]; exact h₁)
    /-
      case mpr.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      h₁ : LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)
      h₂ : LE.le (Neg.neg n) ↑(LieModule.chainBotCoeff (⇑α) β)
      k : Nat
      hk : Eq (↑k) (HSub.hSub (↑(LieModule.chainTopCoeff (⇑α) β)) n)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
    -/
    rw [eq_sub_iff_add_eq, ← eq_sub_iff_add_eq'] at hk
    /-
      case mpr.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      n : Int
      h₁ : LE.le n ↑(LieModule.chainTopCoeff (⇑α) β)
      h₂ : LE.le (Neg.neg n) ↑(LieModule.chainBotCoeff (⇑α) β)
      k : Nat
      hk : Eq n (HSub.hSub ↑(LieModule.chainTopCoeff (⇑α) β) ↑k)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot
    -/
    subst hk
    simp only [neg_sub, tsub_le_iff_right, ← Nat.cast_add, Nat.cast_le,
      chainBotCoeff_add_chainTopCoeff] at h₂
    /-
      case mpr.intro.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      k : Nat
      h₁ : LE.le (HSub.hSub ↑(LieModule.chainTopCoeff (⇑α) β) ↑k) ↑(LieModule.chainT …
      h₂ : LE.le k (LieAlgebra.IsKilling.chainLength α β)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (HSub.hSub ↑(LieModule.ch …
    -/
    have := rootSpace_neg_nsmul_add_chainTop_of_le α β h₂
    rwa [coe_chainTop, ← Nat.cast_smul_eq_nsmul ℤ, ← neg_smul,
      ← add_assoc, ← add_smul, ← sub_eq_neg_add] at this


lemma rootSpace_zsmul_add_ne_bot_iff_mem (hα : α.IsNonZero) (n : ℤ) :
    rootSpace H (n • α + β) ≠ ⊥ ↔ n ∈ Finset.Icc (-chainBotCoeff α β : ℤ) (chainTopCoeff α β) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    n : Int
    ⊢ Iff (Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)) Bot.bot)  …
  -/
  rw [rootSpace_zsmul_add_ne_bot_iff α β hα n, Finset.mem_Icc, and_comm, neg_le]
  /-
    🎉 no goals
  -/


lemma chainTopCoeff_of_eq_zsmul_add
    (hα : α.IsNonZero) (β' : Weight K H L) (n : ℤ) (hβ' : (β' : H → K) = n • α + β) :
    chainTopCoeff α β' = chainTopCoeff α β - n := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Int
    hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
    ⊢ Eq (↑(LieModule.chainTopCoeff (⇑α) β')) (HSub.hSub (↑(LieModule.chainTopCoef …
  -/
  apply le_antisymm
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ LE.le (↑(LieModule.chainTopCoeff (⇑α) β')) (HSub.hSub (↑(LieModule.chainTopC …
    -/
  · refine le_sub_iff_add_le.mpr ((rootSpace_zsmul_add_ne_bot_iff α β hα _).mp ?_).1
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (HAdd.hAdd (↑(LieModule.c …
    -/
    rw [add_smul, add_assoc, ← hβ', ← coe_chainTop]
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ Ne (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β')) Bot.bot
    -/
    exact (chainTop α β').2
    /-
      🎉 no goals
    -/
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ LE.le (HSub.hSub (↑(LieModule.chainTopCoeff (⇑α) β)) n) ↑(LieModule.chainTop …
    -/
  · refine ((rootSpace_zsmul_add_ne_bot_iff α β' hα _).mp ?_).1
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ Ne (LieAlgebra.rootSpace H (HAdd.hAdd (HSMul.hSMul (HSub.hSub (↑(LieModule.c …
    -/
    rw [hβ', ← add_assoc, ← add_smul, sub_add_cancel, ← coe_chainTop]
    /-
      case a
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      ⊢ Ne (LieAlgebra.rootSpace H ⇑(LieModule.chainTop (⇑α) β)) Bot.bot
    -/
    exact (chainTop α β).2
    /-
      🎉 no goals
    -/


lemma chainBotCoeff_of_eq_zsmul_add
    (hα : α.IsNonZero) (β' : Weight K H L) (n : ℤ) (hβ' : (β' : H → K) = n • α + β) :
    chainBotCoeff α β' = chainBotCoeff α β + n := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Int
    hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
    ⊢ Eq (↑(LieModule.chainBotCoeff (⇑α) β')) (HAdd.hAdd (↑(LieModule.chainBotCoef …
  -/
  have : (β' : H → K) = -n • (-α) + β := by rwa [neg_smul, smul_neg, neg_neg]
  rw [chainBotCoeff, chainBotCoeff, ← Weight.coe_neg,
    chainTopCoeff_of_eq_zsmul_add (-α) β hα.neg β' (-n) this, sub_neg_eq_add]


lemma chainLength_of_eq_zsmul_add (β' : Weight K H L) (n : ℤ) (hβ' : (β' : H → K) = n • α + β) :
    chainLength α β' = chainLength α β := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    n : Int
    hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
    ⊢ Eq (LieAlgebra.IsKilling.chainLength α β') (LieAlgebra.IsKilling.chainLength …
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      hα : α.IsZero
      ⊢ Eq (LieAlgebra.IsKilling.chainLength α β') (LieAlgebra.IsKilling.chainLength …
    -/
  · rw [chainLength_of_isZero _ _ hα, chainLength_of_isZero _ _ hα]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β β' : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      n : Int
      hβ' : Eq (⇑β') (HAdd.hAdd (HSMul.hSMul n ⇑α) ⇑β)
      hα : Not α.IsZero
      ⊢ Eq (LieAlgebra.IsKilling.chainLength α β') (LieAlgebra.IsKilling.chainLength …
    -/
  · apply Nat.cast_injective (R := ℤ)
    rw [← chainTopCoeff_add_chainBotCoeff, ← chainTopCoeff_add_chainBotCoeff,
      Nat.cast_add, Nat.cast_add, chainTopCoeff_of_eq_zsmul_add α β hα β' n hβ',
      chainBotCoeff_of_eq_zsmul_add α β hα β' n hβ', sub_eq_add_neg, add_add_add_comm,
      neg_add_cancel, add_zero]


lemma chainTopCoeff_zero_right [Nontrivial L] (hα : α.IsNonZero) :
    chainTopCoeff α (0 : Weight K H L) = 1 := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    ⊢ Eq (LieModule.chainTopCoeff (⇑α) 0) 1
  -/
  symm
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    ⊢ Eq 1 (LieModule.chainTopCoeff (⇑α) 0)
  -/
  apply eq_of_le_of_not_lt
    /-
      case hab
      K : Type u_1
      L : Type u_2
      inst✝⁸ : Field K
      inst✝⁷ : CharZero K
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : LieAlgebra.IsKilling K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      inst✝ : Nontrivial L
      hα : α.IsNonZero
      ⊢ LE.le 1 (LieModule.chainTopCoeff (⇑α) 0)
    -/
  · rw [Nat.one_le_iff_ne_zero]
    /-
      case hab
      K : Type u_1
      L : Type u_2
      inst✝⁸ : Field K
      inst✝⁷ : CharZero K
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : LieAlgebra.IsKilling K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      inst✝ : Nontrivial L
      hα : α.IsNonZero
      ⊢ Ne (LieModule.chainTopCoeff (⇑α) 0) 0
    -/
    intro e
    exact α.2 (by simpa [e, Weight.coe_zero] using
      genWeightSpace_chainTopCoeff_add_one_nsmul_add α (0 : Weight K H L) hα)
  /-
    case hba
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  obtain ⟨x, hx, x_ne0⟩ := (chainTop α (0 : Weight K H L)).exists_ne_zero
  /-
    case hba.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  obtain ⟨h, e, f, isSl2, he, hf⟩ := exists_isSl2Triple_of_weight_isNonZero hα
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    h e f : L
    isSl2 : IsSl2Triple h e f
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  obtain rfl := isSl2.h_eq_coroot hα he hf
  have prim : isSl2.HasPrimitiveVectorWith x (chainLength α (0 : Weight K H L) : K) :=
    have := lie_mem_genWeightSpace_of_mem_genWeightSpace he hx
    ⟨x_ne0, (chainLength_smul _ _ hx).symm, by rwa [genWeightSpace_add_chainTop _ _ hα] at this⟩
  obtain ⟨k, hk⟩ : ∃ k : K, k • f =
      (toEnd K L L f ^ (chainTopCoeff α (0 : Weight K H L) + 1)) x := by
    have : (toEnd K L L f ^ (chainTopCoeff α (0 : Weight K H L) + 1)) x ∈ rootSpace H (-α) := by
      convert toEnd_pow_apply_mem hf hx (chainTopCoeff α (0 : Weight K H L) + 1) using 2
      rw [coe_chainTop', Weight.coe_zero, add_zero, succ_nsmul',
        add_assoc, smul_neg, neg_add_cancel, add_zero]
    simpa using (finrank_eq_one_iff_of_nonzero' ⟨f, hf⟩ (by simpa using isSl2.f_ne_zero)).mp
      (finrank_rootSpace_eq_one _ hα.neg) ⟨_, this⟩
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq (HSMul.hSMul k f) ((HPow.hPow ((LieModule.toEnd K L L) f) (HAdd.hAdd ( …
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  apply_fun (⁅f, ·⁆) at hk
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq (Bracket.bracket f (HSMul.hSMul k f)) (Bracket.bracket f ((HPow.hPow ( …
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  simp only [lie_smul, lie_self, smul_zero, prim.lie_f_pow_toEnd_f] at hk
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq 0 ((HPow.hPow ((LieModule.toEnd K L L) f) (HAdd.hAdd (HAdd.hAdd (LieMo …
    ⊢ Not (LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0))
  -/
  intro e
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e✝ f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e✝
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e✝ f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq 0 ((HPow.hPow ((LieModule.toEnd K L L) f) (HAdd.hAdd (HAdd.hAdd (LieMo …
    e : LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0)
    ⊢ False
  -/
  refine prim.pow_toEnd_f_ne_zero_of_eq_nat rfl ?_ hk.symm
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e✝ f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e✝
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e✝ f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq 0 ((HPow.hPow ((LieModule.toEnd K L L) f) (HAdd.hAdd (HAdd.hAdd (LieMo …
    e : LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (LieModule.chainTopCoeff (⇑α) 0) 1) 1) (LieAlgeb …
  -/
  have := (apply_coroot_eq_cast' α 0).symm
  simp only [← @Nat.cast_two ℤ, ← Nat.cast_mul, Weight.zero_apply, Int.cast_eq_zero, sub_eq_zero,
    Nat.cast_inj] at this
  /-
    case hba.intro.intro.intro.intro.intro.intro.intro.intro
    K : Type u_1
    L : Type u_2
    inst✝⁸ : Field K
    inst✝⁷ : CharZero K
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : LieAlgebra.IsKilling K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    inst✝ : Nontrivial L
    hα : α.IsNonZero
    x : L
    hx : Membership.mem (LieModule.genWeightSpace L ⇑(LieModule.chainTop (⇑α) 0)) x
    x_ne0 : Ne x 0
    e✝ f : L
    he : Membership.mem (LieAlgebra.rootSpace H ⇑α) e✝
    hf : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    isSl2 : IsSl2Triple (↑(LieAlgebra.IsKilling.coroot α)) e✝ f
    prim : isSl2.HasPrimitiveVectorWith x ↑(LieAlgebra.IsKilling.chainLength α 0)
    k : K
    hk : Eq 0 ((HPow.hPow ((LieModule.toEnd K L L) f) (HAdd.hAdd (HAdd.hAdd (LieMo …
    e : LT.lt 1 (LieModule.chainTopCoeff (⇑α) 0)
    this : Eq (LieAlgebra.IsKilling.chainLength α 0) (HMul.hMul 2 (LieModule.chain …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (LieModule.chainTopCoeff (⇑α) 0) 1) 1) (LieAlgeb …
  -/
  rwa [this, Nat.succ_le, two_mul, add_lt_add_iff_left]
  /-
    🎉 no goals
  -/


lemma chainBotCoeff_zero_right [Nontrivial L] (hα : α.IsNonZero) :
    chainBotCoeff α (0 : Weight K H L) = 1 :=
  chainTopCoeff_zero_right (-α) hα.neg


lemma chainLength_zero_right [Nontrivial L] (hα : α.IsNonZero) : chainLength α 0 = 2 := by
  rw [← chainBotCoeff_add_chainTopCoeff, chainTopCoeff_zero_right α hα,
    chainBotCoeff_zero_right α hα]


lemma rootSpace_two_smul (hα : α.IsNonZero) : rootSpace H (2 • α) = ⊥ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Eq (LieAlgebra.rootSpace H (HSMul.hSMul 2 ⇑α)) Bot.bot
  -/
  cases subsingleton_or_nontrivial L
    /-
      case inl
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      h✝ : Subsingleton L
      ⊢ Eq (LieAlgebra.rootSpace H (HSMul.hSMul 2 ⇑α)) Bot.bot
    -/
  · exact IsEmpty.elim inferInstance α
    /-
      🎉 no goals
    -/
  simpa [chainTopCoeff_zero_right α hα] using
    genWeightSpace_chainTopCoeff_add_one_nsmul_add α (0 : Weight K H L) hα


lemma rootSpace_one_div_two_smul (hα : α.IsNonZero) : rootSpace H ((2⁻¹ : K) • α) = ⊥ := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Eq (LieAlgebra.rootSpace H (HSMul.hSMul (Inv.inv 2) ⇑α)) Bot.bot
  -/
  by_contra h
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    h : Not (Eq (LieAlgebra.rootSpace H (HSMul.hSMul (Inv.inv 2) ⇑α)) Bot.bot)
    ⊢ False
  -/
  let W : Weight K H L := ⟨_, h⟩
  have hW : 2 • (W : H → K) = α := by
    show 2 • (2⁻¹ : K) • (α : H → K) = α
    rw [← Nat.cast_smul_eq_nsmul K, smul_smul]; simp
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    h : Not (Eq (LieAlgebra.rootSpace H (HSMul.hSMul (Inv.inv 2) ⇑α)) Bot.bot)
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    hW : Eq (HSMul.hSMul 2 ⇑W) ⇑α
    ⊢ False
  -/
  apply α.genWeightSpace_ne_bot
  have := rootSpace_two_smul W (fun (e : (W : H → K) = 0) ↦ hα <| by
    apply_fun (2 • ·) at e; simpa [hW] using e)
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    h : Not (Eq (LieAlgebra.rootSpace H (HSMul.hSMul (Inv.inv 2) ⇑α)) Bot.bot)
    W : LieModule.Weight K (Subtype fun x => Membership.mem H x) L := { toFun := H …
    hW : Eq (HSMul.hSMul 2 ⇑W) ⇑α
    this : Eq (LieAlgebra.rootSpace H (HSMul.hSMul 2 ⇑W)) Bot.bot
    ⊢ Eq (LieModule.genWeightSpace L ⇑α) Bot.bot
  -/
  rwa [hW] at this
  /-
    🎉 no goals
  -/


lemma eq_neg_one_or_eq_zero_or_eq_one_of_eq_smul
    (hα : α.IsNonZero) (k : K) (h : (β : H → K) = k • α) :
    k = -1 ∨ k = 0 ∨ k = 1 := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
  -/
  cases subsingleton_or_nontrivial L
    /-
      case inl
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Subsingleton L
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
  · exact IsEmpty.elim inferInstance α
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    h✝ : Nontrivial L
    ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
  -/
  have H := apply_coroot_eq_cast' α β
  /-
    case inr
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H✝ : LieSubalgebra K L
    inst✝¹ : H✝.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
    hα : α.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    h✝ : Nontrivial L
    H : Eq (β (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub (↑(LieAlgebra.IsKilling …
    ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
  -/
  rw [h] at H
  /-
    case inr
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H✝ : LieSubalgebra K L
    inst✝¹ : H✝.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
    hα : α.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    h✝ : Nontrivial L
    H : Eq (HSMul.hSMul k (⇑α) (LieAlgebra.IsKilling.coroot α)) ↑(HSub.hSub (↑(Lie …
    ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
  -/
  simp only [Pi.smul_apply, root_apply_coroot hα] at H
  /-
    case inr
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H✝ : LieSubalgebra K L
    inst✝¹ : H✝.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
    hα : α.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    h✝ : Nontrivial L
    H : Eq (HSMul.hSMul k 2) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.chainLength α β)) …
    ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
  -/
  rcases (chainLength α β).even_or_odd with (⟨n, hn⟩|⟨n, hn⟩)
    /-
      case inr.inl.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      H : Eq (HSMul.hSMul k 2) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.chainLength α β)) …
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n n)
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
  · rw [hn, ← two_mul] at H
    simp only [smul_eq_mul, Nat.cast_mul, Nat.cast_ofNat, ← mul_sub, ← mul_comm (2 : K),
      Int.cast_sub, Int.cast_mul, Int.cast_ofNat, Int.cast_natCast,
      mul_eq_mul_left_iff, OfNat.ofNat_ne_zero, or_false] at H
    /-
      case inr.inl.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n n)
      H : Eq k (HSub.hSub ↑n ↑(LieModule.chainTopCoeff (⇑α) β))
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    rw [← Int.cast_natCast, ← Int.cast_natCast (chainTopCoeff α β), ← Int.cast_sub] at H
    have := (rootSpace_zsmul_add_ne_bot_iff_mem α 0 hα (n - chainTopCoeff α β)).mp
      (by rw [← Int.cast_smul_eq_zsmul K, ← H, ← h, Weight.coe_zero, add_zero]; exact β.2)
    /-
      case inr.inl.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n n)
      H : Eq k ↑(HSub.hSub ↑n ↑(LieModule.chainTopCoeff (⇑α) β))
      this : Membership.mem (Finset.Icc (Neg.neg ↑(LieModule.chainBotCoeff (⇑α) 0))  …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    rw [chainTopCoeff_zero_right α hα, chainBotCoeff_zero_right α hα, Nat.cast_one] at this
    /-
      case inr.inl.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n n)
      H : Eq k ↑(HSub.hSub ↑n ↑(LieModule.chainTopCoeff (⇑α) β))
      this : Membership.mem (Finset.Icc (-1) 1) (HSub.hSub ↑n ↑(LieModule.chainTopCo …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    set k' : ℤ := n - chainTopCoeff α β
    /-
      case inr.inl.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd n n)
      k' : Int := HSub.hSub ↑n ↑(LieModule.chainTopCoeff (⇑α) β)
      H : Eq k ↑k'
      this : Membership.mem (Finset.Icc (-1) 1) k'
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    subst H
    have : k' ∈ ({-1, 0, 1} : Finset ℤ) := by
      show k' ∈ Finset.Icc (-1 : ℤ) (1 : ℤ)
      exact this
    simpa only [Int.reduceNeg, Finset.mem_insert, Finset.mem_singleton, ← @Int.cast_inj K,
      Int.cast_zero, Int.cast_neg, Int.cast_one] using this
    /-
      case inr.inr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      H : Eq (HSMul.hSMul k 2) ↑(HSub.hSub (↑(LieAlgebra.IsKilling.chainLength α β)) …
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
  · apply_fun (· / 2) at H
    /-
      case inr.inr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HSMul.hSMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(LieAlgebra.I …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    rw [hn, smul_eq_mul] at H
    /-
      case inr.inr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    have hk : k = n + 2⁻¹ - chainTopCoeff α β := by simpa [sub_div, add_div] using H
    /-
      case inr.inr.intro
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
      hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    have := (rootSpace_zsmul_add_ne_bot_iff α β hα (chainTopCoeff α β - n)).mpr ?_
    /-
      case inr.inr.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
      hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
      this : Ne (LieAlgebra.rootSpace H✝ (HAdd.hAdd (HSMul.hSMul (HSub.hSub ↑(LieMod …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    swap
      /-
        case inr.inr.intro.refine_1
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H✝ : LieSubalgebra K L
        inst✝¹ : H✝.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
        α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
        hα : α.IsNonZero
        k : K
        h : Eq (⇑β) (HSMul.hSMul k ⇑α)
        h✝ : Nontrivial L
        n : Nat
        hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
        H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
        hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
        ⊢ And (LE.le (HSub.hSub ↑(LieModule.chainTopCoeff (⇑α) β) ↑n) ↑(LieModule.chai …
      -/
    · simp only [tsub_le_iff_right, le_add_iff_nonneg_right, Nat.cast_nonneg, neg_sub, true_and]
      /-
        case inr.inr.intro.refine_1
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H✝ : LieSubalgebra K L
        inst✝¹ : H✝.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
        α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
        hα : α.IsNonZero
        k : K
        h : Eq (⇑β) (HSMul.hSMul k ⇑α)
        h✝ : Nontrivial L
        n : Nat
        hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
        H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
        hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
        ⊢ LE.le (↑n) (HAdd.hAdd ↑(LieModule.chainBotCoeff (⇑α) β) ↑(LieModule.chainTop …
      -/
      rw [← Nat.cast_add, chainBotCoeff_add_chainTopCoeff, hn]
      /-
        case inr.inr.intro.refine_1
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H✝ : LieSubalgebra K L
        inst✝¹ : H✝.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
        α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
        hα : α.IsNonZero
        k : K
        h : Eq (⇑β) (HSMul.hSMul k ⇑α)
        h✝ : Nontrivial L
        n : Nat
        hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
        H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
        hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
        ⊢ LE.le ↑n ↑(HAdd.hAdd (HMul.hMul 2 n) 1)
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
      hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
      this : Ne (LieAlgebra.rootSpace H✝ (HAdd.hAdd (HSMul.hSMul (HSub.hSub ↑(LieMod …
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    rw [h, hk, ← Int.cast_smul_eq_zsmul K, ← add_smul] at this
    simp only [Int.cast_sub, Int.cast_natCast,
      sub_add_sub_cancel', add_sub_cancel_left, ne_eq] at this
    /-
      case inr.inr.intro.refine_2
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H✝ : LieSubalgebra K L
      inst✝¹ : H✝.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H✝ x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H✝ x) L
      hα : α.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      h✝ : Nontrivial L
      n : Nat
      hn : Eq (LieAlgebra.IsKilling.chainLength α β) (HAdd.hAdd (HMul.hMul 2 n) 1)
      H : Eq (HDiv.hDiv (HMul.hMul k 2) 2) (HDiv.hDiv (↑(HSub.hSub (↑(HAdd.hAdd (HMu …
      hk : Eq k (HSub.hSub (HAdd.hAdd (↑n) (Inv.inv 2)) ↑(LieModule.chainTopCoeff (⇑ …
      this : Not (Eq (LieAlgebra.rootSpace H✝ (HSMul.hSMul (Inv.inv 2) ⇑α)) Bot.bot)
      ⊢ Or (Eq k (-1)) (Or (Eq k 0) (Eq k 1))
    -/
    cases this (rootSpace_one_div_two_smul α hα)
    /-
      🎉 no goals
    -/


/-- `±α` are the only `K`-multiples of a root `α` that are also (non-zero) roots. -/
lemma eq_neg_or_eq_of_eq_smul (hβ : β.IsNonZero) (k : K) (h : (β : H → K) = k • α) :
    β = -α ∨ β = α := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : β.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
  -/
  by_cases hα : α.IsZero
    /-
      case pos
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : β.IsNonZero
      k : K
      h : Eq (⇑β) (HSMul.hSMul k ⇑α)
      hα : α.IsZero
      ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
    -/
  · rw [hα, smul_zero] at h; cases hβ h
                             /-
                               🎉 no goals
                             -/
  /-
    case neg
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : β.IsNonZero
    k : K
    h : Eq (⇑β) (HSMul.hSMul k ⇑α)
    hα : Not α.IsZero
    ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
  -/
  rcases eq_neg_one_or_eq_zero_or_eq_one_of_eq_smul α β hα k h with (rfl | rfl | rfl)
    /-
      case neg.inl
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : β.IsNonZero
      hα : Not α.IsZero
      h : Eq (⇑β) (HSMul.hSMul (-1) ⇑α)
      ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
    -/
  · exact .inl (by ext; rw [h, neg_one_smul]; rfl)
    /-
      🎉 no goals
    -/
    /-
      case neg.inr.inl
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : β.IsNonZero
      hα : Not α.IsZero
      h : Eq (⇑β) (HSMul.hSMul 0 ⇑α)
      ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
    -/
  · cases hβ (by rwa [zero_smul] at h)
    /-
      🎉 no goals
    -/
    /-
      case neg.inr.inr
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : β.IsNonZero
      hα : Not α.IsZero
      h : Eq (⇑β) (HSMul.hSMul 1 ⇑α)
      ⊢ Or (Eq β (Neg.neg α)) (Eq β α)
    -/
  · exact .inr (by ext; rw [h, one_smul])
    /-
      🎉 no goals
    -/


/-- The reflection of a root along another. -/
def reflectRoot (α β : Weight K H L) : Weight K H L where
  toFun := β - β (coroot α) • α
  genWeightSpace_ne_bot' := by
    /-
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α✝ β✝ α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      ⊢ Ne (LieModule.genWeightSpace L (HSub.hSub (⇑β) (HSMul.hSMul (β (LieAlgebra.I …
    -/
    by_cases hα : α.IsZero
      /-
        case pos
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ β✝ α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : α.IsZero
        ⊢ Ne (LieModule.genWeightSpace L (HSub.hSub (⇑β) (HSMul.hSMul (β (LieAlgebra.I …
      -/
    · simpa [hα.eq] using β.genWeightSpace_ne_bot
      /-
        🎉 no goals
      -/
    rw [sub_eq_neg_add, apply_coroot_eq_cast α β, ← neg_smul, ← Int.cast_neg,
      Int.cast_smul_eq_zsmul, rootSpace_zsmul_add_ne_bot_iff α β hα]
    /-
      case neg
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α✝ β✝ α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Not α.IsZero
      ⊢ And (LE.le (Neg.neg (HSub.hSub ↑(LieModule.chainBotCoeff (⇑α) β) ↑(LieModule …
    -/
    omega
    /-
      🎉 no goals
    -/


lemma reflectRoot_isNonZero (α β : Weight K H L) (hβ : β.IsNonZero) :
    (reflectRoot α β).IsNonZero := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : β.IsNonZero
    ⊢ (LieAlgebra.IsKilling.reflectRoot α β).IsNonZero
  -/
  intro e
  have : β (coroot α) = 0 := by
    by_cases hα : α.IsZero
    · simp [coroot_eq_zero_iff.mpr hα]
    apply add_left_injective (β (coroot α))
    simpa [root_apply_coroot hα, mul_two] using congr_fun (sub_eq_zero.mp e) (coroot α)
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : β.IsNonZero
    e : (LieAlgebra.IsKilling.reflectRoot α β).IsZero
    this : Eq (β (LieAlgebra.IsKilling.coroot α)) 0
    ⊢ False
  -/
  have : reflectRoot α β = β := by ext; simp [reflectRoot, this]
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : β.IsNonZero
    e : (LieAlgebra.IsKilling.reflectRoot α β).IsZero
    this✝ : Eq (β (LieAlgebra.IsKilling.coroot α)) 0
    this : Eq (LieAlgebra.IsKilling.reflectRoot α β) β
    ⊢ False
  -/
  exact hβ (this ▸ e)
  /-
    🎉 no goals
  -/


/-- The root system of a finite-dimensional Lie algebra with non-degenerate Killing form over a
field of characteristic zero, relative to a splitting Cartan subalgebra. -/
def rootSystem :
    RootSystem H.root K (Dual K H) H :=
  RootSystem.mk'
    IsReflexive.toPerfectPairingDual
    { toFun := (↑)
      inj' := by
        /-
          K : Type u_1
          L : Type u_2
          inst✝⁷ : Field K
          inst✝⁶ : CharZero K
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra K L
          inst✝³ : LieAlgebra.IsKilling K L
          inst✝² : FiniteDimensional K L
          H : LieSubalgebra K L
          inst✝¹ : H.IsCartanSubalgebra
          inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
          α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
          ⊢ Function.Injective fun x => LieModule.Weight.toLinear K (Subtype fun x => Me …
        -/
        intro α β h; ext x; simpa using LinearMap.congr_fun h x  }
                            /-
                              🎉 no goals
                            -/
    { toFun := coroot ∘ (↑)
                 /-
                   K : Type u_1
                   L : Type u_2
                   inst✝⁷ : Field K
                   inst✝⁶ : CharZero K
                   inst✝⁵ : LieRing L
                   inst✝⁴ : LieAlgebra K L
                   inst✝³ : LieAlgebra.IsKilling K L
                   inst✝² : FiniteDimensional K L
                   H : LieSubalgebra K L
                   inst✝¹ : H.IsCartanSubalgebra
                   inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                   α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                   ⊢ Function.Injective (Function.comp LieAlgebra.IsKilling.coroot Subtype.val)
                 -/
      inj' := by rintro ⟨α, hα⟩ ⟨β, hβ⟩ h; simpa using h }
                                           /-
                                             🎉 no goals
                                           -/
                      /-
                        K : Type u_1
                        L : Type u_2
                        inst✝⁷ : Field K
                        inst✝⁶ : CharZero K
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra K L
                        inst✝³ : LieAlgebra.IsKilling K L
                        inst✝² : FiniteDimensional K L
                        H : LieSubalgebra K L
                        inst✝¹ : H.IsCartanSubalgebra
                        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                        α✝ β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                        x✝ : Subtype fun x => Membership.mem LieSubalgebra.root x
                        α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                        hα : Membership.mem LieSubalgebra.root α
                        ⊢ Eq ((IsReflexive.toPerfectPairingDual.toLin ({ toFun := fun x => LieModule.W …
                      -/
    (fun ⟨α, hα⟩ ↦ by simpa using root_apply_coroot <| by simpa using hα)
                      /-
                        🎉 no goals
                      -/
    (by
      /-
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        ⊢ ∀ (i : Subtype fun x => Membership.mem LieSubalgebra.root x), Set.MapsTo (⇑( …
      -/
      rintro ⟨α, hα⟩ - ⟨⟨β, hβ⟩, rfl⟩
      simp only [Function.Embedding.coeFn_mk, IsReflexive.toPerfectPairingDual_toLin,
        Function.comp_apply, Set.mem_range, Subtype.exists, exists_prop]
      /-
        case mk.intro.mk
        K : Type u_1
        L : Type u_2
        inst✝⁷ : Field K
        inst✝⁶ : CharZero K
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra K L
        inst✝³ : LieAlgebra.IsKilling K L
        inst✝² : FiniteDimensional K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ β✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : Membership.mem LieSubalgebra.root α
        β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hβ : Membership.mem LieSubalgebra.root β
        ⊢ Exists fun a => And (Membership.mem LieSubalgebra.root a) (Eq (LieModule.Wei …
      -/
      exact ⟨reflectRoot α β, (by simpa using reflectRoot_isNonZero α β <| by simpa using hβ), rfl⟩)
      /-
        🎉 no goals
      -/
        /-
          K : Type u_1
          L : Type u_2
          inst✝⁷ : Field K
          inst✝⁶ : CharZero K
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra K L
          inst✝³ : LieAlgebra.IsKilling K L
          inst✝² : FiniteDimensional K L
          H : LieSubalgebra K L
          inst✝¹ : H.IsCartanSubalgebra
          inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
          α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
          ⊢ Eq (Submodule.span K (Set.range ⇑{ toFun := fun x => LieModule.Weight.toLine …
        -/
    (by convert span_weight_isNonZero_eq_top K L H; ext; simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
lemma corootForm_rootSystem_eq_killing :
    (rootSystem H).CorootForm = (killingForm K L).restrict H := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (LieAlgebra.IsKilling.rootSystem H).CorootForm ((killingForm K L).restric …
  -/
  rw [restrict_killingForm_eq_sum, RootPairing.CorootForm, ← Finset.sum_coe_sort (s := H.root)]
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (Finset.univ.sum fun i => LinearMap.smulRight ((LieAlgebra.IsKilling.root …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma rootSystem_toPerfectPairing_apply (f x) : (rootSystem H).toPerfectPairing f x = f x :=
  rfl

@[deprecated (since := "2024-09-09")]
alias rootSystem_toLin_apply := rootSystem_toPerfectPairing_apply

@[simp] lemma rootSystem_pairing_apply (α β) : (rootSystem H).pairing β α = β.1 (coroot α.1) := rfl

@[simp] lemma rootSystem_root_apply (α) : (rootSystem H).root α = α := rfl

@[simp] lemma rootSystem_coroot_apply (α) : (rootSystem H).coroot α = coroot α := rfl


instance : (rootSystem H).IsCrystallographic where
  exists_int α β :=
                                                       /-
                                                         K : Type u_1
                                                         L : Type u_2
                                                         inst✝⁷ : Field K
                                                         inst✝⁶ : CharZero K
                                                         inst✝⁵ : LieRing L
                                                         inst✝⁴ : LieAlgebra K L
                                                         inst✝³ : LieAlgebra.IsKilling K L
                                                         inst✝² : FiniteDimensional K L
                                                         H : LieSubalgebra K L
                                                         inst✝¹ : H.IsCartanSubalgebra
                                                         inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                                                         α✝ β✝ : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                                                         α β : Subtype fun x => Membership.mem LieSubalgebra.root x
                                                         ⊢ Eq (↑(HSub.hSub ↑(LieModule.chainBotCoeff ⇑↑β ↑α) ↑(LieModule.chainTopCoeff  …
                                                       -/
    ⟨chainBotCoeff β.1 α.1 - chainTopCoeff β.1 α.1, by simp [apply_coroot_eq_cast β.1 α.1]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem isReduced_rootSystem : (rootSystem H).IsReduced := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ (LieAlgebra.IsKilling.rootSystem H).IsReduced
  -/
  intro ⟨α, hα⟩ ⟨β, hβ⟩ e
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Membership.mem LieSubalgebra.root α
    β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : Membership.mem LieSubalgebra.root β
    e : Not (LinearIndependent K (Matrix.vecCons ((LieAlgebra.IsKilling.rootSystem …
    ⊢ Or (Eq ((LieAlgebra.IsKilling.rootSystem H).root ⟨α, hα⟩) ((LieAlgebra.IsKil …
  -/
  rw [LinearIndependent.pair_iff' ((rootSystem H).ne_zero _), not_forall] at e
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Membership.mem LieSubalgebra.root α
    β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : Membership.mem LieSubalgebra.root β
    e : Exists fun x => Not (Ne (HSMul.hSMul x ((LieAlgebra.IsKilling.rootSystem H …
    ⊢ Or (Eq ((LieAlgebra.IsKilling.rootSystem H).root ⟨α, hα⟩) ((LieAlgebra.IsKil …
  -/
  simp only [Nat.succ_eq_add_one, Nat.reduceAdd, rootSystem_root_apply, ne_eq, not_not] at e
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁷ : Field K
    inst✝⁶ : CharZero K
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra K L
    inst✝³ : LieAlgebra.IsKilling K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Membership.mem LieSubalgebra.root α
    β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hβ : Membership.mem LieSubalgebra.root β
    e : Exists fun x => Eq (HSMul.hSMul x (LieModule.Weight.toLinear K (Subtype fu …
    ⊢ Or (Eq ((LieAlgebra.IsKilling.rootSystem H).root ⟨α, hα⟩) ((LieAlgebra.IsKil …
  -/
  obtain ⟨u, hu⟩ := e
  obtain (h | h) :=
    eq_neg_or_eq_of_eq_smul α β (by simpa using hβ) u (by ext x; exact DFunLike.congr_fun hu.symm x)
    /-
      case intro.inl
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Membership.mem LieSubalgebra.root α
      β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : Membership.mem LieSubalgebra.root β
      u : K
      hu : Eq (HSMul.hSMul u (LieModule.Weight.toLinear K (Subtype fun x => Membersh …
      h : Eq β (Neg.neg α)
      ⊢ Or (Eq ((LieAlgebra.IsKilling.rootSystem H).root ⟨α, hα⟩) ((LieAlgebra.IsKil …
    -/
  · right; ext x; simpa [neg_eq_iff_eq_neg] using DFunLike.congr_fun h.symm x
                  /-
                    🎉 no goals
                  -/
    /-
      case intro.inr
      K : Type u_1
      L : Type u_2
      inst✝⁷ : Field K
      inst✝⁶ : CharZero K
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra K L
      inst✝³ : LieAlgebra.IsKilling K L
      inst✝² : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Membership.mem LieSubalgebra.root α
      β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hβ : Membership.mem LieSubalgebra.root β
      u : K
      hu : Eq (HSMul.hSMul u (LieModule.Weight.toLinear K (Subtype fun x => Membersh …
      h : Eq β α
      ⊢ Or (Eq ((LieAlgebra.IsKilling.rootSystem H).root ⟨α, hα⟩) ((LieAlgebra.IsKil …
    -/
  · left; ext x; simpa using DFunLike.congr_fun h.symm x
                 /-
                   🎉 no goals
                 -/


