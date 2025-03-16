/-- **Fundamental theorem of algebra**: every non constant complex polynomial
  has a root -/
theorem exists_root {f : ℂ[X]} (hf : 0 < degree f) : ∃ z : ℂ, IsRoot f z := by
  /-
    f : Polynomial Complex
    hf : LT.lt 0 f.degree
    ⊢ Exists fun z => f.IsRoot z
  -/
  by_contra! hf'
  /- Since `f` has no roots, `f⁻¹` is differentiable. And since `f` is a polynomial, it tends to
  infinity at infinity, thus `f⁻¹` tends to zero at infinity. By Liouville's theorem, `f⁻¹ = 0`. -/
  have (z : ℂ) : (f.eval z)⁻¹ = 0 :=
    (f.differentiable.inv hf').apply_eq_of_tendsto_cocompact z <|
      Metric.cobounded_eq_cocompact (α := ℂ) ▸ (Filter.tendsto_inv₀_cobounded.comp <| by
        simpa only [tendsto_norm_atTop_iff_cobounded]
          using f.tendsto_norm_atTop hf tendsto_norm_cobounded_atTop)
  -- Thus `f = 0`, contradicting the fact that `0 < degree f`.
  /-
    f : Polynomial Complex
    hf : LT.lt 0 f.degree
    hf' : ∀ (z : Complex), Not (f.IsRoot z)
    this : ∀ (z : Complex), Eq (Inv.inv (Polynomial.eval z f)) 0
    ⊢ False
  -/
  obtain rfl : f = C 0 := Polynomial.funext fun z ↦ inv_injective <| by simp [this]
  /-
    hf : LT.lt 0 (Polynomial.C 0).degree
    hf' : ∀ (z : Complex), Not ((Polynomial.C 0).IsRoot z)
    this : ∀ (z : Complex), Eq (Inv.inv (Polynomial.eval z (Polynomial.C 0))) 0
    ⊢ False
  -/
  simp at hf
  /-
    🎉 no goals
  -/


instance isAlgClosed : IsAlgClosed ℂ :=
  IsAlgClosed.of_exists_root _ fun _p _ hp => Complex.exists_root <| degree_pos_of_irreducible hp


theorem splits_ℚ_ℂ {p : ℚ[X]} : Fact (p.Splits (algebraMap ℚ ℂ)) :=
  ⟨IsAlgClosed.splits_codomain p⟩


/-- The number of complex roots equals the number of real roots plus
    the number of roots not fixed by complex conjugation (i.e. with some imaginary component). -/
theorem card_complex_roots_eq_card_real_add_card_not_gal_inv (p : ℚ[X]) :
    (p.rootSet ℂ).toFinset.card =
      (p.rootSet ℝ).toFinset.card +
        (galActionHom p ℂ (restrict p ℂ
        (AlgEquiv.restrictScalars ℚ Complex.conjAe))).support.card := by
  /-
    p : Polynomial Rat
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (p.rootSet Real).toFinset.ca …
  -/
  by_cases hp : p = 0
    /-
      case pos
      p : Polynomial Rat
      hp : Eq p 0
      ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (p.rootSet Real).toFinset.ca …
    -/
  · haveI : IsEmpty (p.rootSet ℂ) := by rw [hp, rootSet_zero]; infer_instance
    simp_rw [(galActionHom p ℂ _).support.eq_empty_of_isEmpty, hp, rootSet_zero,
      Set.toFinset_empty, Finset.card_empty]
  /-
    case neg
    p : Polynomial Rat
    hp : Not (Eq p 0)
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (p.rootSet Real).toFinset.ca …
  -/
  have inj : Function.Injective (IsScalarTower.toAlgHom ℚ ℝ ℂ) := (algebraMap ℝ ℂ).injective
  rw [← Finset.card_image_of_injective _ Subtype.coe_injective, ←
    Finset.card_image_of_injective _ inj]
  /-
    case neg
    p : Polynomial Rat
    hp : Not (Eq p 0)
    inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (Finset.image (⇑(IsScalarTow …
  -/
  let a : Finset ℂ := ?_
  /-
    case neg.refine_2
    p : Polynomial Rat
    hp : Not (Eq p 0)
    inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
    a : Finset Complex := ?neg.refine_1✝
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (Finset.image (⇑(IsScalarTow …
  -/
  on_goal 1 => let b : Finset ℂ := ?_
  /-
    case neg.refine_2.refine_2
    p : Polynomial Rat
    hp : Not (Eq p 0)
    inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
    a : Finset Complex := ?neg.refine_1✝
    b : Finset Complex := ?neg.refine_2.refine_1✝
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (Finset.image (⇑(IsScalarTow …
  -/
  on_goal 1 => let c : Finset ℂ := ?_
  -- Porting note: was
  --   change a.card = b.card + c.card
  /-
    case neg.refine_2.refine_2.refine_2
    p : Polynomial Rat
    hp : Not (Eq p 0)
    inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
    a : Finset Complex := ?neg.refine_1✝
    b : Finset Complex := ?neg.refine_2.refine_1✝
    c : Finset Complex := ?neg.refine_2.refine_2.refine_1✝
    ⊢ Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (Finset.image (⇑(IsScalarTow …
  -/
  suffices a.card = b.card + c.card by exact this
  have ha : ∀ z : ℂ, z ∈ a ↔ aeval z p = 0 := by
    intro z; rw [Set.mem_toFinset, mem_rootSet_of_ne hp]
  have hb : ∀ z : ℂ, z ∈ b ↔ aeval z p = 0 ∧ z.im = 0 := by
    intro z
    simp_rw [b, Finset.mem_image, Set.mem_toFinset, mem_rootSet_of_ne hp]
    constructor
    · rintro ⟨w, hw, rfl⟩
      exact ⟨by rw [aeval_algHom_apply, hw, map_zero], rfl⟩
    · rintro ⟨hz1, hz2⟩
      have key : IsScalarTower.toAlgHom ℚ ℝ ℂ z.re = z := by
        ext
        · rfl
        · rw [hz2]; rfl
      exact ⟨z.re, inj (by rwa [← aeval_algHom_apply, key, map_zero]), key⟩
  have hc0 :
    ∀ w : p.rootSet ℂ, galActionHom p ℂ (restrict p ℂ (Complex.conjAe.restrictScalars ℚ)) w = w ↔
        w.val.im = 0 := by
    intro w
    rw [Subtype.ext_iff, galActionHom_restrict]
    exact Complex.conj_eq_iff_im
  have hc : ∀ z : ℂ, z ∈ c ↔ aeval z p = 0 ∧ z.im ≠ 0 := by
    intro z
    simp_rw [c, Finset.mem_image]
    constructor
    · rintro ⟨w, hw, rfl⟩
      exact ⟨(mem_rootSet.mp w.2).2, mt (hc0 w).mpr (Equiv.Perm.mem_support.mp hw)⟩
    · rintro ⟨hz1, hz2⟩
      exact ⟨⟨z, mem_rootSet.mpr ⟨hp, hz1⟩⟩, Equiv.Perm.mem_support.mpr (mt (hc0 _).mp hz2), rfl⟩
  /-
    case neg.refine_2.refine_2.refine_2
    p : Polynomial Rat
    hp : Not (Eq p 0)
    inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
    a : Finset Complex := (p.rootSet Complex).toFinset
    b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
    c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
    ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
    hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
    hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
    hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
    ⊢ Eq a.card (HAdd.hAdd b.card c.card)
  -/
  rw [← Finset.card_union_of_disjoint]
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      ⊢ Eq a.card (Union.union b c).card
    -/
  · apply congr_arg Finset.card
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      ⊢ Eq a (Union.union b c)
    -/
    simp_rw [Finset.ext_iff, Finset.mem_union, ha, hb, hc]
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      ⊢ ∀ (a : Complex), Iff (Eq ((Polynomial.aeval a) p) 0) (Or (And (Eq ((Polynomi …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      ⊢ Disjoint b c
    -/
  · rw [Finset.disjoint_left]
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      ⊢ ∀ ⦃a : Complex⦄, Membership.mem b a → Not (Membership.mem c a)
    -/
    intro z
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      z : Complex
      ⊢ Membership.mem b z → Not (Membership.mem c z)
    -/
    rw [hb, hc]
    /-
      case neg.refine_2.refine_2.refine_2
      p : Polynomial Rat
      hp : Not (Eq p 0)
      inj : Function.Injective ⇑(IsScalarTower.toAlgHom Rat Real Complex)
      a : Finset Complex := (p.rootSet Complex).toFinset
      b : Finset Complex := Finset.image (⇑(IsScalarTower.toAlgHom Rat Real Complex) …
      c : Finset Complex := Finset.image (fun a => ↑a) ((Polynomial.Gal.galActionHom …
      ha : ∀ (z : Complex), Iff (Membership.mem a z) (Eq ((Polynomial.aeval z) p) 0)
      hb : ∀ (z : Complex), Iff (Membership.mem b z) (And (Eq ((Polynomial.aeval z)  …
      hc0 : ∀ (w : ↑(p.rootSet Complex)), Iff (Eq (((Polynomial.Gal.galActionHom p C …
      hc : ∀ (z : Complex), Iff (Membership.mem c z) (And (Eq ((Polynomial.aeval z)  …
      z : Complex
      ⊢ And (Eq ((Polynomial.aeval z) p) 0) (Eq z.im 0) → Not (And (Eq ((Polynomial. …
    -/
    tauto
    /-
      🎉 no goals
    -/


/-- An irreducible polynomial of prime degree with two non-real roots has full Galois group. -/
theorem galActionHom_bijective_of_prime_degree {p : ℚ[X]} (p_irr : Irreducible p)
    (p_deg : p.natDegree.Prime)
    (p_roots : Fintype.card (p.rootSet ℂ) = Fintype.card (p.rootSet ℝ) + 2) :
    Function.Bijective (galActionHom p ℂ) := by
  classical
  have h1 : Fintype.card (p.rootSet ℂ) = p.natDegree := by
    simp_rw [rootSet_def, Finset.coe_sort_coe, Fintype.card_coe]
    rw [Multiset.toFinset_card_of_nodup, ← natDegree_eq_card_roots]
    · exact IsAlgClosed.splits_codomain p
    · exact nodup_roots ((separable_map (algebraMap ℚ ℂ)).mpr p_irr.separable)
  let conj' := restrict p ℂ (Complex.conjAe.restrictScalars ℚ)
  refine
    ⟨galActionHom_injective p ℂ, fun x =>
      (congr_arg (x ∈ ·) (show (galActionHom p ℂ).range = ⊤ from ?_)).mpr
        (Subgroup.mem_top x)⟩
  apply Equiv.Perm.subgroup_eq_top_of_swap_mem
  · rwa [h1]
  · rw [h1]
    simpa only [Fintype.card_eq_nat_card,
      Nat.card_congr (MonoidHom.ofInjective (galActionHom_injective p ℂ)).toEquiv.symm]
      using prime_degree_dvd_card p_irr p_deg
  · exact ⟨conj', rfl⟩
  · rw [← Equiv.Perm.card_support_eq_two]
    apply Nat.add_left_cancel
    rw [← p_roots, ← Set.toFinset_card (rootSet p ℝ), ← Set.toFinset_card (rootSet p ℂ)]
    exact (card_complex_roots_eq_card_real_add_card_not_gal_inv p).symm


/-- An irreducible polynomial of prime degree with 1-3 non-real roots has full Galois group. -/
theorem galActionHom_bijective_of_prime_degree' {p : ℚ[X]} (p_irr : Irreducible p)
    (p_deg : p.natDegree.Prime)
    (p_roots1 : Fintype.card (p.rootSet ℝ) + 1 ≤ Fintype.card (p.rootSet ℂ))
    (p_roots2 : Fintype.card (p.rootSet ℂ) ≤ Fintype.card (p.rootSet ℝ) + 3) :
    Function.Bijective (galActionHom p ℂ) := by
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le (HAdd.hAdd (Fintype.card ↑(p.rootSet Real)) 1) (Fintype.card  …
    p_roots2 : LE.le (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card  …
    ⊢ Function.Bijective ⇑(Polynomial.Gal.galActionHom p Complex)
  -/
  apply galActionHom_bijective_of_prime_degree p_irr p_deg
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le (HAdd.hAdd (Fintype.card ↑(p.rootSet Real)) 1) (Fintype.card  …
    p_roots2 : LE.le (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card  …
    ⊢ Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.rootSet  …
  -/
  let n := (galActionHom p ℂ (restrict p ℂ (Complex.conjAe.restrictScalars ℚ))).support.card
  have hn : 2 ∣ n :=
    Equiv.Perm.two_dvd_card_support
      (by
         rw [← MonoidHom.map_pow, ← MonoidHom.map_pow,
          show AlgEquiv.restrictScalars ℚ Complex.conjAe ^ 2 = 1 from
            AlgEquiv.ext Complex.conj_conj,
          MonoidHom.map_one, MonoidHom.map_one])
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le (HAdd.hAdd (Fintype.card ↑(p.rootSet Real)) 1) (Fintype.card  …
    p_roots2 : LE.le (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card  …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    ⊢ Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.rootSet  …
  -/
  have key := card_complex_roots_eq_card_real_add_card_not_gal_inv p
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le (HAdd.hAdd (Fintype.card ↑(p.rootSet Real)) 1) (Fintype.card  …
    p_roots2 : LE.le (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card  …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    key : Eq (p.rootSet Complex).toFinset.card (HAdd.hAdd (p.rootSet Real).toFinse …
    ⊢ Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.rootSet  …
  -/
  simp_rw [Set.toFinset_card] at key
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le (HAdd.hAdd (Fintype.card ↑(p.rootSet Real)) 1) (Fintype.card  …
    p_roots2 : LE.le (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card  …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    key : Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.root …
    ⊢ Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.rootSet  …
  -/
  rw [key, add_le_add_iff_left] at p_roots1 p_roots2
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le 1 ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.r …
    p_roots2 : LE.le ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.res …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    key : Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.root …
    ⊢ Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.rootSet  …
  -/
  rw [key, add_right_inj]
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le 1 ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.r …
    p_roots2 : LE.le ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.res …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    key : Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.root …
    ⊢ Eq ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict p Comp …
  -/
  suffices ∀ m : ℕ, 2 ∣ m → 1 ≤ m → m ≤ 3 → m = 2 by exact this n hn p_roots1 p_roots2
  /-
    p : Polynomial Rat
    p_irr : Irreducible p
    p_deg : Nat.Prime p.natDegree
    p_roots1 : LE.le 1 ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.r …
    p_roots2 : LE.le ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.res …
    n : Nat := ((Polynomial.Gal.galActionHom p Complex) ((Polynomial.Gal.restrict  …
    hn : Dvd.dvd 2 n
    key : Eq (Fintype.card ↑(p.rootSet Complex)) (HAdd.hAdd (Fintype.card ↑(p.root …
    ⊢ ∀ (m : Nat), Dvd.dvd 2 m → LE.le 1 m → LE.le m 3 → Eq m 2
  -/
  rintro m ⟨k, rfl⟩ h2 h3
  exact le_antisymm
      (Nat.lt_succ_iff.mp
        (lt_of_le_of_ne h3 (show 2 * k ≠ 2 * 1 + 1 from Nat.two_mul_ne_two_mul_add_one)))
      (Nat.succ_le_iff.mpr
        (lt_of_le_of_ne h2 (show 2 * 0 + 1 ≠ 2 * k from Nat.two_mul_ne_two_mul_add_one.symm)))


lemma Polynomial.mul_star_dvd_of_aeval_eq_zero_im_ne_zero (p : ℝ[X]) {z : ℂ} (h0 : aeval z p = 0)
    (hz : z.im ≠ 0) : (X - C ((starRingEnd ℂ) z)) * (X - C z) ∣ map (algebraMap ℝ ℂ) p := by
  /-
    p : Polynomial Real
    z : Complex
    h0 : Eq ((Polynomial.aeval z) p) 0
    hz : Ne z.im 0
    ⊢ Dvd.dvd (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C ((starRingEnd Compl …
  -/
  apply IsCoprime.mul_dvd
    /-
      case H
      p : Polynomial Real
      z : Complex
      h0 : Eq ((Polynomial.aeval z) p) 0
      hz : Ne z.im 0
      ⊢ IsCoprime (HSub.hSub Polynomial.X (Polynomial.C ((starRingEnd Complex) z)))  …
    -/
  · exact isCoprime_X_sub_C_of_isUnit_sub <| .mk0 _ <| sub_ne_zero.2 <| mt conj_eq_iff_im.1 hz
    /-
      🎉 no goals
    -/
    /-
      case H1
      p : Polynomial Real
      z : Complex
      h0 : Eq ((Polynomial.aeval z) p) 0
      hz : Ne z.im 0
      ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C ((starRingEnd Complex) z))) (P …
    -/
  · simpa [dvd_iff_isRoot, aeval_conj]
    /-
      🎉 no goals
    -/
    /-
      case H2
      p : Polynomial Real
      z : Complex
      h0 : Eq ((Polynomial.aeval z) p) 0
      hz : Ne z.im 0
      ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C z)) (Polynomial.map (algebraMa …
    -/
  · simpa [dvd_iff_isRoot]
    /-
      🎉 no goals
    -/


/-- If `z` is a non-real complex root of a real polynomial,
then `p` is divisible by a quadratic polynomial. -/
lemma Polynomial.quadratic_dvd_of_aeval_eq_zero_im_ne_zero (p : ℝ[X]) {z : ℂ} (h0 : aeval z p = 0)
    (hz : z.im ≠ 0) : X ^ 2 - C (2 * z.re) * X + C (‖z‖ ^ 2) ∣ p := by
  /-
    p : Polynomial Real
    z : Complex
    h0 : Eq ((Polynomial.aeval z) p) 0
    hz : Ne z.im 0
    ⊢ Dvd.dvd (HAdd.hAdd (HSub.hSub (HPow.hPow Polynomial.X 2) (HMul.hMul (Polynom …
  -/
  rw [← map_dvd_map' (algebraMap ℝ ℂ)]
  /-
    p : Polynomial Real
    z : Complex
    h0 : Eq ((Polynomial.aeval z) p) 0
    hz : Ne z.im 0
    ⊢ Dvd.dvd (Polynomial.map (algebraMap Real Complex) (HAdd.hAdd (HSub.hSub (HPo …
  -/
  convert p.mul_star_dvd_of_aeval_eq_zero_im_ne_zero h0 hz
  calc
    map (algebraMap ℝ ℂ) (X ^ 2 - C (2 * z.re) * X + C (‖z‖ ^ 2))
    _ = X ^ 2 - C (↑(2 * z.re) : ℂ) * X + C (‖z‖ ^ 2 : ℂ) := by simp
    _ = (X - C (conj z)) * (X - C z) := by
      rw [← add_conj, map_add, ← mul_conj', map_mul]
      ring


/-- An irreducible real polynomial has degree at most two. -/
lemma Irreducible.degree_le_two {p : ℝ[X]} (hp : Irreducible p) : degree p ≤ 2 := by
  obtain ⟨z, hz⟩ : ∃ z : ℂ, aeval z p = 0 :=
    IsAlgClosed.exists_aeval_eq_zero _ p (degree_pos_of_irreducible hp).ne'
  cases eq_or_ne z.im 0 with
  | inl hz0 =>
    lift z to ℝ using hz0
    erw [aeval_ofReal, RCLike.ofReal_eq_zero] at hz
    exact (degree_eq_one_of_irreducible_of_root hp hz).trans_le one_le_two
  | inr hz0 =>
    obtain ⟨q, rfl⟩ := p.quadratic_dvd_of_aeval_eq_zero_im_ne_zero hz hz0
    have hd : degree (X ^ 2 - C (2 * z.re) * X + C (‖z‖ ^ 2)) = 2 := by
      compute_degree!
    have hq : IsUnit q := by
      refine (of_irreducible_mul hp).resolve_left (mt isUnit_iff_degree_eq_zero.1 ?_)
      rw [hd]
      exact two_ne_zero
    refine (degree_mul_le _ _).trans_eq ?_
    rwa [isUnit_iff_degree_eq_zero.1 hq, add_zero]


/-- An irreducible real polynomial has natural degree at most two. -/
lemma Irreducible.natDegree_le_two {p : ℝ[X]} (hp : Irreducible p) : natDegree p ≤ 2 :=
  natDegree_le_iff_degree_le.2 hp.degree_le_two


@[deprecated (since := "2024-02-18")]
alias Irreducible.nat_degree_le_two := Irreducible.natDegree_le_two

