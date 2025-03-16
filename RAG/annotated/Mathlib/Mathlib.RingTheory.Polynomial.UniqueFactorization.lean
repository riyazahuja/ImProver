instance (priority := 100) wfDvdMonoid : WfDvdMonoid R[X] where
  wf := by
    classical
      refine
        RelHomClass.wellFounded
          (⟨fun p : R[X] =>
              ((if p = 0 then ⊤ else ↑p.degree : WithTop (WithBot ℕ)), p.leadingCoeff), ?_⟩ :
            DvdNotUnit →r Prod.Lex (· < ·) DvdNotUnit)
          (wellFounded_lt.prod_lex ‹WfDvdMonoid R›.wf)
      rintro a b ⟨ane0, ⟨c, ⟨not_unit_c, rfl⟩⟩⟩
      dsimp
      rw [Polynomial.degree_mul, if_neg ane0]
      split_ifs with hac
      · rw [hac, Polynomial.leadingCoeff_zero]
        apply Prod.Lex.left
        exact WithTop.coe_lt_top _
      have cne0 : c ≠ 0 := right_ne_zero_of_mul hac
      simp only [cne0, ane0, Polynomial.leadingCoeff_mul]
      by_cases hdeg : c.degree = (0 : ℕ)
      · simp only [hdeg, Nat.cast_zero, add_zero]
        refine Prod.Lex.right _ ⟨?_, ⟨c.leadingCoeff, fun unit_c => not_unit_c ?_, rfl⟩⟩
        · rwa [Ne, Polynomial.leadingCoeff_eq_zero]
        rw [Polynomial.isUnit_iff, Polynomial.eq_C_of_degree_eq_zero hdeg]
        use c.leadingCoeff, unit_c
        rw [Polynomial.leadingCoeff, Polynomial.natDegree_eq_of_degree_eq_some hdeg]
      · apply Prod.Lex.left
        rw [Polynomial.degree_eq_natDegree cne0] at *
        simp only [Nat.cast_inj] at hdeg
        rw [WithTop.coe_lt_coe, Polynomial.degree_eq_natDegree ane0, ← Nat.cast_add, Nat.cast_lt]
        exact lt_add_of_pos_right _ (Nat.pos_of_ne_zero hdeg)


theorem exists_irreducible_of_degree_pos (hf : 0 < f.degree) : ∃ g, Irreducible g ∧ g ∣ f :=
  WfDvdMonoid.exists_irreducible_factor (fun huf => ne_of_gt hf <| degree_eq_zero_of_isUnit huf)
    fun hf0 => not_lt_of_lt hf <| hf0.symm ▸ (@degree_zero R _).symm ▸ WithBot.bot_lt_coe _


theorem exists_irreducible_of_natDegree_pos (hf : 0 < f.natDegree) : ∃ g, Irreducible g ∧ g ∣ f :=
  exists_irreducible_of_degree_pos <| by
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : WfDvdMonoid R
      f : Polynomial R
      hf : LT.lt 0 f.natDegree
      ⊢ LT.lt 0 f.degree
    -/
    contrapose! hf
    /-
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : WfDvdMonoid R
      f : Polynomial R
      hf : LE.le f.degree 0
      ⊢ LE.le f.natDegree 0
    -/
    exact natDegree_le_of_degree_le hf
    /-
      🎉 no goals
    -/


theorem exists_irreducible_of_natDegree_ne_zero (hf : f.natDegree ≠ 0) :
    ∃ g, Irreducible g ∧ g ∣ f :=
  exists_irreducible_of_natDegree_pos <| Nat.pos_of_ne_zero hf


instance (priority := 100) uniqueFactorizationMonoid : UniqueFactorizationMonoid D[X] := by
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    ⊢ UniqueFactorizationMonoid (Polynomial D)
  -/
  letI := Classical.arbitrary (NormalizedGCDMonoid D)
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    this : NormalizedGCDMonoid D := Classical.arbitrary (NormalizedGCDMonoid D)
    ⊢ UniqueFactorizationMonoid (Polynomial D)
  -/
  exact ufm_of_decomposition_of_wfDvdMonoid
  /-
    🎉 no goals
  -/


/-- If `D` is a unique factorization domain, `f` is a non-zero polynomial in `D[X]`, then `f` has
only finitely many monic factors.
(Note that its factors up to unit may be more than monic factors.)
See also `UniqueFactorizationMonoid.fintypeSubtypeDvd`. -/
noncomputable def fintypeSubtypeMonicDvd (f : D[X]) (hf : f ≠ 0) :
    Fintype { g : D[X] // g.Monic ∧ g ∣ f } := by
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    ⊢ Fintype (Subtype fun g => And g.Monic (Dvd.dvd g f))
  -/
  set G := { g : D[X] // g.Monic ∧ g ∣ f }
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    ⊢ Fintype G
  -/
  let y : Associates D[X] := Associates.mk f
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y : Associates (Polynomial D) := Associates.mk f
    ⊢ Fintype G
  -/
  have hy : y ≠ 0 := Associates.mk_ne_zero.mpr hf
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y : Associates (Polynomial D) := Associates.mk f
    hy : Ne y 0
    ⊢ Fintype G
  -/
  let H := { x : Associates D[X] // x ∣ y }
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y : Associates (Polynomial D) := Associates.mk f
    hy : Ne y 0
    H : Type (max 0 u) := Subtype fun x => Dvd.dvd x y
    ⊢ Fintype G
  -/
  let hfin : Fintype H := UniqueFactorizationMonoid.fintypeSubtypeDvd y hy
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y : Associates (Polynomial D) := Associates.mk f
    hy : Ne y 0
    H : Type (max 0 u) := Subtype fun x => Dvd.dvd x y
    hfin : Fintype H := UniqueFactorizationMonoid.fintypeSubtypeDvd y hy
    ⊢ Fintype G
  -/
  let i : G → H := fun x ↦ ⟨Associates.mk x.1, Associates.mk_dvd_mk.2 x.2.2⟩
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y : Associates (Polynomial D) := Associates.mk f
    hy : Ne y 0
    H : Type (max 0 u) := Subtype fun x => Dvd.dvd x y
    hfin : Fintype H := UniqueFactorizationMonoid.fintypeSubtypeDvd y hy
    i : G → H := fun x => ⟨Associates.mk ↑x, ⋯⟩
    ⊢ Fintype G
  -/
  refine Fintype.ofInjective i fun x y heq ↦ ?_
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y✝ : Associates (Polynomial D) := Associates.mk f
    hy : Ne y✝ 0
    H : Type (max 0 u) := Subtype fun x => Dvd.dvd x y✝
    hfin : Fintype H := UniqueFactorizationMonoid.fintypeSubtypeDvd y✝ hy
    i : G → H := fun x => ⟨Associates.mk ↑x, ⋯⟩
    x y : G
    heq : Eq (i x) (i y)
    ⊢ Eq x y
  -/
  rw [Subtype.mk.injEq] at heq ⊢
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    f : Polynomial D
    hf : Ne f 0
    G : Type u := Subtype fun g => And g.Monic (Dvd.dvd g f)
    y✝ : Associates (Polynomial D) := Associates.mk f
    hy : Ne y✝ 0
    H : Type (max 0 u) := Subtype fun x => Dvd.dvd x y✝
    hfin : Fintype H := UniqueFactorizationMonoid.fintypeSubtypeDvd y✝ hy
    i : G → H := fun x => ⟨Associates.mk ↑x, ⋯⟩
    x y : G
    heq : Eq (Associates.mk ↑x) (Associates.mk ↑y)
    ⊢ Eq ↑x ↑y
  -/
  exact eq_of_monic_of_associated x.2.1 y.2.1 (Associates.mk_eq_mk_iff_associated.mp heq)
  /-
    🎉 no goals
  -/


private theorem uniqueFactorizationMonoid_of_fintype [Fintype σ] :
    UniqueFactorizationMonoid (MvPolynomial σ D) :=
  (renameEquiv D (Fintype.equivFin σ)).toMulEquiv.symm.uniqueFactorizationMonoid <| by
    /-
      σ : Type v
      D : Type u
      inst✝³ : CommRing D
      inst✝² : IsDomain D
      inst✝¹ : UniqueFactorizationMonoid D
      inst✝ : Fintype σ
      ⊢ UniqueFactorizationMonoid (MvPolynomial (Fin (Fintype.card σ)) D)
    -/
    induction' Fintype.card σ with d hd
      /-
        case zero
        σ : Type v
        D : Type u
        inst✝³ : CommRing D
        inst✝² : IsDomain D
        inst✝¹ : UniqueFactorizationMonoid D
        inst✝ : Fintype σ
        ⊢ UniqueFactorizationMonoid (MvPolynomial (Fin 0) D)
      -/
    · apply (isEmptyAlgEquiv D (Fin 0)).toMulEquiv.symm.uniqueFactorizationMonoid
      /-
        case zero
        σ : Type v
        D : Type u
        inst✝³ : CommRing D
        inst✝² : IsDomain D
        inst✝¹ : UniqueFactorizationMonoid D
        inst✝ : Fintype σ
        ⊢ UniqueFactorizationMonoid D
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case succ
        σ : Type v
        D : Type u
        inst✝³ : CommRing D
        inst✝² : IsDomain D
        inst✝¹ : UniqueFactorizationMonoid D
        inst✝ : Fintype σ
        d : Nat
        hd : UniqueFactorizationMonoid (MvPolynomial (Fin d) D)
        ⊢ UniqueFactorizationMonoid (MvPolynomial (Fin (HAdd.hAdd d 1)) D)
      -/
    · apply (finSuccEquiv D d).toMulEquiv.symm.uniqueFactorizationMonoid
      /-
        case succ
        σ : Type v
        D : Type u
        inst✝³ : CommRing D
        inst✝² : IsDomain D
        inst✝¹ : UniqueFactorizationMonoid D
        inst✝ : Fintype σ
        d : Nat
        hd : UniqueFactorizationMonoid (MvPolynomial (Fin d) D)
        ⊢ UniqueFactorizationMonoid (Polynomial (MvPolynomial (Fin d) D))
      -/
      exact Polynomial.uniqueFactorizationMonoid
      /-
        🎉 no goals
      -/


instance (priority := 100) uniqueFactorizationMonoid :
    UniqueFactorizationMonoid (MvPolynomial σ D) := by
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    d : Nat
    ⊢ UniqueFactorizationMonoid (MvPolynomial σ D)
  -/
  rw [iff_exists_prime_factors]
  /-
    σ : Type v
    D : Type u
    inst✝² : CommRing D
    inst✝¹ : IsDomain D
    inst✝ : UniqueFactorizationMonoid D
    d : Nat
    ⊢ ∀ (a : MvPolynomial σ D), Ne a 0 → Exists fun f => And (∀ (b : MvPolynomial  …
  -/
  intro a ha; obtain ⟨s, a', rfl⟩ := exists_finset_rename a
  obtain ⟨w, h, u, hw⟩ :=
    iff_exists_prime_factors.1 (uniqueFactorizationMonoid_of_fintype s) a' fun h =>
      ha <| by simp [h]
  exact
    ⟨w.map (rename (↑)), fun b hb =>
      let ⟨b', hb', he⟩ := Multiset.mem_map.1 hb
      he ▸ (prime_rename_iff (σ := σ) ↑s).2 (h b' hb'),
      Units.map (@rename s σ D _ (↑)).toRingHom.toMonoidHom u, by
      erw [Multiset.prod_hom, ← map_mul, hw]⟩


/-- A polynomial over a field which is not a unit must have a monic irreducible factor.
See also `WfDvdMonoid.exists_irreducible_factor`. -/
theorem Polynomial.exists_monic_irreducible_factor {F : Type*} [Field F] (f : F[X])
    (hu : ¬IsUnit f) : ∃ g : F[X], g.Monic ∧ Irreducible g ∧ g ∣ f := by
  /-
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    hu : Not (IsUnit f)
    ⊢ Exists fun g => And g.Monic (And (Irreducible g) (Dvd.dvd g f))
  -/
  by_cases hf : f = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      f : Polynomial F
      hu : Not (IsUnit f)
      hf : Eq f 0
      ⊢ Exists fun g => And g.Monic (And (Irreducible g) (Dvd.dvd g f))
    -/
  · exact ⟨X, monic_X, irreducible_X, hf ▸ dvd_zero X⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    hu : Not (IsUnit f)
    hf : Not (Eq f 0)
    ⊢ Exists fun g => And g.Monic (And (Irreducible g) (Dvd.dvd g f))
  -/
  obtain ⟨g, hi, hf⟩ := WfDvdMonoid.exists_irreducible_factor hu hf
  have ha : Associated g (g * C g.leadingCoeff⁻¹) := associated_mul_unit_right _ _ <|
    isUnit_C.2 (leadingCoeff_ne_zero.2 hi.ne_zero).isUnit.inv
  /-
    case neg.intro.intro
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    hu : Not (IsUnit f)
    hf✝ : Not (Eq f 0)
    g : Polynomial F
    hi : Irreducible g
    hf : Dvd.dvd g f
    ha : Associated g (HMul.hMul g (Polynomial.C (Inv.inv g.leadingCoeff)))
    ⊢ Exists fun g => And g.Monic (And (Irreducible g) (Dvd.dvd g f))
  -/
  exact ⟨_, monic_mul_leadingCoeff_inv hi.ne_zero, ha.irreducible hi, ha.dvd_iff_dvd_left.1 hf⟩
  /-
    🎉 no goals
  -/

