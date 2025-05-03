/-- If `A` is a finite free `R`-algebra, then `f : A` is nilpotent on `κ(𝔭) ⊗ A` for some
prime `𝔭 ◃ R` if and only if every non-leading coefficient of `charpoly(f)` is in `𝔭`. -/
lemma isNilpotent_tensor_residueField_iff
    [Module.Free R A] [Module.Finite R A] (f : A) (I : Ideal R) [I.IsPrime] :
    IsNilpotent (algebraMap A (A ⊗[R] I.ResidueField) f) ↔
      ∀ i < Module.finrank R A, (Algebra.lmul R A f).charpoly.coeff i ∈ I := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Module.Free R A
    inst✝¹ : Module.Finite R A
    f : A
    I : Ideal R
    inst✝ : I.IsPrime
    ⊢ Iff (IsNilpotent ((algebraMap A (TensorProduct R A I.ResidueField)) f)) (∀ ( …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Subsingleton R
      ⊢ Iff (IsNilpotent ((algebraMap A (TensorProduct R A I.ResidueField)) f)) (∀ ( …
    -/
  · have := (algebraMap R (A ⊗[R] I.ResidueField)).codomain_trivial
    /-
      case inl
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Subsingleton R
      this : Subsingleton (TensorProduct R A I.ResidueField)
      ⊢ Iff (IsNilpotent ((algebraMap A (TensorProduct R A I.ResidueField)) f)) (∀ ( …
    -/
    simp [Subsingleton.elim I ⊤, Subsingleton.elim (f ⊗ₜ[R] (1 : I.ResidueField)) 0]
    /-
      🎉 no goals
    -/
  have : Module.finrank I.ResidueField (I.ResidueField ⊗[R] A) = Module.finrank R A := by
    rw [Module.finrank_tensorProduct, Module.finrank_self, one_mul]
  /-
    case inr
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Module.Free R A
    inst✝¹ : Module.Finite R A
    f : A
    I : Ideal R
    inst✝ : I.IsPrime
    h✝ : Nontrivial R
    this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
    ⊢ Iff (IsNilpotent ((algebraMap A (TensorProduct R A I.ResidueField)) f)) (∀ ( …
  -/
  rw [← IsNilpotent.map_iff (Algebra.TensorProduct.comm R A I.ResidueField).injective]
  simp only [Algebra.TensorProduct.algebraMap_apply, Algebra.id.map_eq_id, RingHom.id_apply,
    Algebra.coe_lmul_eq_mul, Algebra.TensorProduct.comm_tmul]
  rw [← IsNilpotent.map_iff (Algebra.lmul_injective (R := I.ResidueField)),
    LinearMap.isNilpotent_iff_charpoly, ← Algebra.baseChange_lmul, LinearMap.charpoly_baseChange]
  /-
    case inr
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Module.Free R A
    inst✝¹ : Module.Finite R A
    f : A
    I : Ideal R
    inst✝ : I.IsPrime
    h✝ : Nontrivial R
    this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
    ⊢ Iff (Eq (Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly (( …
  -/
  simp_rw [this, ← ((LinearMap.mul R A) f).charpoly_natDegree]
  /-
    case inr
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : Module.Free R A
    inst✝¹ : Module.Finite R A
    f : A
    I : Ideal R
    inst✝ : I.IsPrime
    h✝ : Nontrivial R
    this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
    ⊢ Iff (Eq (Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly (( …
  -/
  constructor
    /-
      case inr.mp
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      ⊢ Eq (Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Algeb …
    -/
  · intro e i hi
    /-
      case inr.mp
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      e : Eq (Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alg …
      i : Nat
      hi : LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree
      ⊢ Membership.mem I (((LinearMap.mul R A) f).charpoly.coeff i)
    -/
    replace e := congr(($e).coeff i)
    simpa only [Algebra.coe_lmul_eq_mul, coeff_map, coeff_X_pow, hi.ne, ↓reduceIte,
      ← RingHom.mem_ker, Ideal.ker_algebraMap_residueField] using e
    /-
      case inr.mpr
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      ⊢ (∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membershi …
    -/
  · intro H
    /-
      case inr.mpr
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
      ⊢ Eq (Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Algeb …
    -/
    ext i
    /-
      case inr.mpr.a
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
      i : Nat
      ⊢ Eq ((Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alge …
    -/
    obtain (hi | hi) := eq_or_ne i ((LinearMap.mul R A) f).charpoly.natDegree
      /-
        case inr.mpr.a.inl
        R : Type u_1
        A : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Module.Free R A
        inst✝¹ : Module.Finite R A
        f : A
        I : Ideal R
        inst✝ : I.IsPrime
        h✝ : Nontrivial R
        this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
        H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
        i : Nat
        hi : Eq i ((LinearMap.mul R A) f).charpoly.natDegree
        ⊢ Eq ((Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alge …
      -/
    · simp only [Algebra.coe_lmul_eq_mul, hi, coeff_map, coeff_X_pow, ↓reduceIte]
      /-
        case inr.mpr.a.inl
        R : Type u_1
        A : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Module.Free R A
        inst✝¹ : Module.Finite R A
        f : A
        I : Ideal R
        inst✝ : I.IsPrime
        h✝ : Nontrivial R
        this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
        H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
        i : Nat
        hi : Eq i ((LinearMap.mul R A) f).charpoly.natDegree
        ⊢ Eq ((algebraMap R I.ResidueField) (((LinearMap.mul R A) f).charpoly.coeff (( …
      -/
      rw [← Polynomial.leadingCoeff, ((LinearMap.mul R A) f).charpoly_monic, map_one]
      /-
        🎉 no goals
      -/
    /-
      case inr.mpr.a.inr
      R : Type u_1
      A : Type u_2
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : Module.Free R A
      inst✝¹ : Module.Finite R A
      f : A
      I : Ideal R
      inst✝ : I.IsPrime
      h✝ : Nontrivial R
      this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
      H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
      i : Nat
      hi : Ne i ((LinearMap.mul R A) f).charpoly.natDegree
      ⊢ Eq ((Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alge …
    -/
    obtain (hi | hi) := lt_or_gt_of_ne hi
      /-
        case inr.mpr.a.inr.inl
        R : Type u_1
        A : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Module.Free R A
        inst✝¹ : Module.Finite R A
        f : A
        I : Ideal R
        inst✝ : I.IsPrime
        h✝ : Nontrivial R
        this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
        H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
        i : Nat
        hi✝ : Ne i ((LinearMap.mul R A) f).charpoly.natDegree
        hi : LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree
        ⊢ Eq ((Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alge …
      -/
    · simpa [hi.ne, ← RingHom.mem_ker, Ideal.ker_algebraMap_residueField] using H i hi
      /-
        🎉 no goals
      -/
      /-
        case inr.mpr.a.inr.inr
        R : Type u_1
        A : Type u_2
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        inst✝² : Module.Free R A
        inst✝¹ : Module.Finite R A
        f : A
        I : Ideal R
        inst✝ : I.IsPrime
        h✝ : Nontrivial R
        this : Eq (Module.finrank I.ResidueField (TensorProduct R I.ResidueField A)) ( …
        H : ∀ (i : Nat), LT.lt i ((LinearMap.mul R A) f).charpoly.natDegree → Membersh …
        i : Nat
        hi✝ : Ne i ((LinearMap.mul R A) f).charpoly.natDegree
        hi : GT.gt i ((LinearMap.mul R A) f).charpoly.natDegree
        ⊢ Eq ((Polynomial.map (algebraMap R I.ResidueField) (LinearMap.charpoly ((Alge …
      -/
    · simp [hi.ne', coeff_eq_zero_of_natDegree_lt hi]
      /-
        🎉 no goals
      -/


/-- Let `A` be an `R`-algebra.
`𝔭 : Spec R` is in the image of `Z(I) ∩ D(f) ⊆ Spec S`
if and only if `f` is not nilpotent on `κ(𝔭) ⊗ A ⧸ I`. -/
lemma mem_image_comap_zeroLocus_sdiff (f : A) (s : Set A) (x) :
    x ∈ comap (algebraMap R A) '' (zeroLocus s \ zeroLocus {f}) ↔
      ¬ IsNilpotent (algebraMap A ((A ⧸ Ideal.span s) ⊗[R] x.asIdeal.ResidueField) f) := by
  /-
    R : Type u_2
    A : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : A
    s : Set A
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Set.image (⇑(PrimeSpectrum.comap (algebraMap R A))) (SD …
  -/
  constructor
    /-
      case mp
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap (algebraMap R A))) (SDiff.s …
    -/
  · rintro ⟨q, ⟨hqg, hqf⟩, rfl⟩ H
    /-
      case mp.intro.intro.intro
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      q : PrimeSpectrum A
      hqg : Membership.mem (PrimeSpectrum.zeroLocus s) q
      hqf : Not (Membership.mem (PrimeSpectrum.zeroLocus (Singleton.singleton f)) q)
      H : IsNilpotent ((algebraMap A (TensorProduct R (HasQuotient.Quotient A (Ideal …
      ⊢ False
    -/
    simp only [mem_zeroLocus, Set.singleton_subset_iff, SetLike.mem_coe] at hqg hqf
    have hs : Ideal.span s ≤ RingHom.ker (algebraMap A q.asIdeal.ResidueField) := by
      rwa [Ideal.span_le, Ideal.ker_algebraMap_residueField]
    let F : (A ⧸ Ideal.span s) ⊗[R] (q.asIdeal.comap (algebraMap R A)).ResidueField →ₐ[A]
        q.asIdeal.ResidueField :=
      Algebra.TensorProduct.lift
        (Ideal.Quotient.liftₐ (Ideal.span s) (Algebra.ofId A _) hs)
        (Ideal.ResidueField.mapₐ _ _ rfl)
        fun _ _ ↦ .all _ _
    /-
      case mp.intro.intro.intro
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      q : PrimeSpectrum A
      H : IsNilpotent ((algebraMap A (TensorProduct R (HasQuotient.Quotient A (Ideal …
      hqg : HasSubset.Subset s ↑q.asIdeal
      hqf : Not (Membership.mem q.asIdeal f)
      hs : LE.le (Ideal.span s) (RingHom.ker (algebraMap A q.asIdeal.ResidueField))
      F : AlgHom A (TensorProduct R (HasQuotient.Quotient A (Ideal.span s)) (Ideal.c …
      ⊢ False
    -/
    have := H.map F
    rw [AlgHom.commutes, isNilpotent_iff_eq_zero, ← RingHom.mem_ker,
      Ideal.ker_algebraMap_residueField] at this
    /-
      case mp.intro.intro.intro
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      q : PrimeSpectrum A
      H : IsNilpotent ((algebraMap A (TensorProduct R (HasQuotient.Quotient A (Ideal …
      hqg : HasSubset.Subset s ↑q.asIdeal
      hqf : Not (Membership.mem q.asIdeal f)
      hs : LE.le (Ideal.span s) (RingHom.ker (algebraMap A q.asIdeal.ResidueField))
      F : AlgHom A (TensorProduct R (HasQuotient.Quotient A (Ideal.span s)) (Ideal.c …
      this : Membership.mem q.asIdeal f
      ⊢ False
    -/
    exact hqf this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      ⊢ Not (IsNilpotent ((algebraMap A (TensorProduct R (HasQuotient.Quotient A (Id …
    -/
  · intro H
    /-
      case mpr
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      H : Not (IsNilpotent ((algebraMap A (TensorProduct R (HasQuotient.Quotient A ( …
      ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap (algebraMap R A))) (SDiff.s …
    -/
    rw [← mem_nilradical, nilradical_eq_sInf, Ideal.mem_sInf] at H
    simp only [Set.mem_setOf_eq, Algebra.TensorProduct.algebraMap_apply,
      Ideal.Quotient.algebraMap_eq, not_forall, Classical.not_imp] at H
    /-
      case mpr
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      H : Exists fun x_1 => Exists fun x_2 => Not (Membership.mem x_1 (TensorProduct …
      ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap (algebraMap R A))) (SDiff.s …
    -/
    obtain ⟨q, hq, hfq⟩ := H
    have : ∀ a ∈ s, Ideal.Quotient.mk (Ideal.span s) a ⊗ₜ[R] 1 ∈ q := fun a ha ↦ by
      simp [Ideal.Quotient.eq_zero_iff_mem.mpr (Ideal.subset_span ha)]
    /-
      case mpr.intro.intro
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      q : Ideal (TensorProduct R (HasQuotient.Quotient A (Ideal.span s)) x.asIdeal.R …
      hq : q.IsPrime
      hfq : Not (Membership.mem q (TensorProduct.tmul R ((Ideal.Quotient.mk (Ideal.s …
      this : ∀ (a : A), Membership.mem s a → Membership.mem q (TensorProduct.tmul R  …
      ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap (algebraMap R A))) (SDiff.s …
    -/
    refine ⟨comap (algebraMap A _) ⟨q, hq⟩, ⟨by simpa [Set.subset_def], by simpa⟩, ?_⟩
    rw [← comap_comp_apply, ← IsScalarTower.algebraMap_eq,
      ← Algebra.TensorProduct.includeRight.comp_algebraMap, comap_comp_apply,
      Subsingleton.elim (α := PrimeSpectrum x.asIdeal.ResidueField) (comap _ _) ⊥]
    /-
      case mpr.intro.intro
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      q : Ideal (TensorProduct R (HasQuotient.Quotient A (Ideal.span s)) x.asIdeal.R …
      hq : q.IsPrime
      hfq : Not (Membership.mem q (TensorProduct.tmul R ((Ideal.Quotient.mk (Ideal.s …
      this : ∀ (a : A), Membership.mem s a → Membership.mem q (TensorProduct.tmul R  …
      ⊢ Eq ((PrimeSpectrum.comap (algebraMap R x.asIdeal.ResidueField)) Bot.bot) x
    -/
    ext a
    /-
      case mpr.intro.intro.asIdeal.h
      R : Type u_2
      A : Type u_1
      inst✝² : CommRing R
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      f : A
      s : Set A
      x : PrimeSpectrum R
      q : Ideal (TensorProduct R (HasQuotient.Quotient A (Ideal.span s)) x.asIdeal.R …
      hq : q.IsPrime
      hfq : Not (Membership.mem q (TensorProduct.tmul R ((Ideal.Quotient.mk (Ideal.s …
      this : ∀ (a : A), Membership.mem s a → Membership.mem q (TensorProduct.tmul R  …
      a : R
      ⊢ Iff (Membership.mem ((PrimeSpectrum.comap (algebraMap R x.asIdeal.ResidueFie …
    -/
    exact congr(a ∈ $(Ideal.ker_algebraMap_residueField _))
    /-
      🎉 no goals
    -/


/-- Let `A` be an `R`-algebra.
`𝔭 : Spec R` is in the image of `D(f) ⊆ Spec S`
if and only if `f` is not nilpotent on `κ(𝔭) ⊗ A`. -/
lemma mem_image_comap_basicOpen (f : A) (x) :
    x ∈ comap (algebraMap R A) '' basicOpen f ↔
      ¬ IsNilpotent (algebraMap A (A ⊗[R] x.asIdeal.ResidueField) f) := by
  have e : A ⊗[R] x.asIdeal.ResidueField ≃ₐ[A]
      (A ⧸ (Ideal.span ∅ : Ideal A)) ⊗[R] x.asIdeal.ResidueField := by
    refine Algebra.TensorProduct.congr ?f AlgEquiv.refl
    rw [Ideal.span_empty]
    exact { __ := (RingEquiv.quotientBot A).symm, __ := Algebra.ofId _ _ }
  rw [← IsNilpotent.map_iff e.injective, AlgEquiv.commutes,
    ← mem_image_comap_zeroLocus_sdiff f ∅ x, zeroLocus_empty, ← Set.compl_eq_univ_diff,
    basicOpen_eq_zeroLocus_compl]


/-- Let `A` be an `R`-algebra. If `A ⧸ I` is finite free over `R`,
then the image of `Z(I) ∩ D(f) ⊆ Spec S` in `Spec R` is compact open. -/
lemma exists_image_comap_of_finite_of_free (f : A) (s : Set A)
    [Module.Finite R (A ⧸ Ideal.span s)] [Module.Free R (A ⧸ Ideal.span s)] :
    ∃ t : Finset R, comap (algebraMap R A) '' (zeroLocus s \ zeroLocus {f}) = (zeroLocus t)ᶜ := by
  classical
  use (Finset.range (Module.finrank R (A ⧸ Ideal.span s))).image
    (Algebra.lmul R (A ⧸ Ideal.span s) (Ideal.Quotient.mk _ f)).charpoly.coeff
  ext x
  rw [mem_image_comap_zeroLocus_sdiff, IsScalarTower.algebraMap_apply A (A ⧸ Ideal.span s),
    isNilpotent_tensor_residueField_iff]
  simp [Set.subset_def]


lemma mem_image_comap_C_basicOpen (f : R[X]) (x : PrimeSpectrum R) :
    x ∈ comap C '' basicOpen f ↔ ∃ i, f.coeff i ∉ x.asIdeal := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : Polynomial R
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Set.image ⇑(PrimeSpectrum.comap Polynomial.C) ↑(PrimeSp …
  -/
  trans f.map (algebraMap R x.asIdeal.ResidueField) ≠ 0
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      ⊢ Iff (Membership.mem (Set.image ⇑(PrimeSpectrum.comap Polynomial.C) ↑(PrimeSp …
    -/
  · refine (mem_image_comap_basicOpen _ _).trans (not_iff_not.mpr ?_)
    let e : R[X] ⊗[R] x.asIdeal.ResidueField ≃ₐ[R] x.asIdeal.ResidueField[X] :=
      (Algebra.TensorProduct.comm R _ _).trans (polyEquivTensor R x.asIdeal.ResidueField).symm
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
      ⊢ Iff (IsNilpotent ((algebraMap (Polynomial R) (TensorProduct R (Polynomial R) …
    -/
    rw [← IsNilpotent.map_iff e.injective, isNilpotent_iff_eq_zero]
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
      ⊢ Iff (Eq (e ((algebraMap (Polynomial R) (TensorProduct R (Polynomial R) x.asI …
    -/
    show (e.toAlgHom.toRingHom).comp (algebraMap _ _) f = 0 ↔ Polynomial.mapRingHom _ f = 0
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
      ⊢ Iff (Eq (((↑e).comp (algebraMap (Polynomial R) (TensorProduct R (Polynomial  …
    -/
    congr!
    /-
      case a.h.e'_2.h.e'_5
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
      ⊢ Eq ((↑e).comp (algebraMap (Polynomial R) (TensorProduct R (Polynomial R) x.a …
    -/
    ext1
      /-
        case a.h.e'_2.h.e'_5.h₁
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        x : PrimeSpectrum R
        e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
        ⊢ Eq (((↑e).comp (algebraMap (Polynomial R) (TensorProduct R (Polynomial R) x. …
      -/
    · ext; simp [e]
           /-
             🎉 no goals
           -/
      /-
        case a.h.e'_2.h.e'_5.h₂
        R : Type u_1
        inst✝ : CommRing R
        f : Polynomial R
        x : PrimeSpectrum R
        e : AlgEquiv R (TensorProduct R (Polynomial R) x.asIdeal.ResidueField) (Polyno …
        ⊢ Eq (((↑e).comp (algebraMap (Polynomial R) (TensorProduct R (Polynomial R) x. …
      -/
    · simp [e, monomial_one_one_eq_X]
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      ⊢ Iff (Ne (Polynomial.map (algebraMap R x.asIdeal.ResidueField) f) 0) (Exists  …
    -/
  · simp [Polynomial.ext_iff]
    /-
      🎉 no goals
    -/


lemma image_comap_C_basicOpen (f : R[X]) :
      comap C '' basicOpen f = (zeroLocus (Set.range f.coeff))ᶜ := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      ⊢ Eq (Set.image ⇑(PrimeSpectrum.comap Polynomial.C) ↑(PrimeSpectrum.basicOpen  …
    -/
    ext p
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      p : PrimeSpectrum R
      ⊢ Iff (Membership.mem (Set.image ⇑(PrimeSpectrum.comap Polynomial.C) ↑(PrimeSp …
    -/
    rw [mem_image_comap_C_basicOpen]
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      p : PrimeSpectrum R
      ⊢ Iff (Exists fun i => Not (Membership.mem p.asIdeal (f.coeff i))) (Membership …
    -/
    simp [Set.range_subset_iff]
    /-
      🎉 no goals
    -/


lemma isOpenMap_comap_C : IsOpenMap (comap (R := R) C) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ IsOpenMap ⇑(PrimeSpectrum.comap Polynomial.C)
  -/
  intro U hU
  /-
    R : Type u_1
    inst✝ : CommRing R
    U : Set (PrimeSpectrum (Polynomial R))
    hU : IsOpen U
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) U)
  -/
  obtain ⟨S, hS, rfl⟩ := isTopologicalBasis_basic_opens.open_eq_sUnion hU
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) S.sUnion)
  -/
  rw [Set.image_sUnion]
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ IsOpen (Set.image (Set.image ⇑(PrimeSpectrum.comap Polynomial.C)) S).sUnion
  -/
  apply isOpen_sUnion
  /-
    case intro.intro.h
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ ∀ (t : Set (PrimeSpectrum R)), Membership.mem (Set.image (Set.image ⇑(PrimeS …
  -/
  rintro _ ⟨t, ht, rfl⟩
  /-
    case intro.intro.h.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    t : Set (PrimeSpectrum (Polynomial R))
    ht : Membership.mem S t
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) t)
  -/
  obtain ⟨r, rfl⟩ := hS ht
  /-
    case intro.intro.h.intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    r : Polynomial R
    ht : Membership.mem S ((fun r => ↑(PrimeSpectrum.basicOpen r)) r)
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) ((fun r => ↑(PrimeSp …
  -/
  simp only [image_comap_C_basicOpen]
  /-
    case intro.intro.h.intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    S : Set (Set (PrimeSpectrum (Polynomial R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    r : Polynomial R
    ht : Membership.mem S ((fun r => ↑(PrimeSpectrum.basicOpen r)) r)
    ⊢ IsOpen (HasCompl.compl (PrimeSpectrum.zeroLocus (Set.range r.coeff)))
  -/
  exact (isClosed_zeroLocus _).isOpen_compl
  /-
    🎉 no goals
  -/


lemma exists_image_comap_of_monic (f g : R[X]) (hg : g.Monic) :
    ∃ t : Finset R, comap C '' (zeroLocus {g} \ zeroLocus {f}) = (zeroLocus t)ᶜ := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    ⊢ Exists fun t => Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.s …
  -/
  apply (config := { allowSynthFailures := true }) exists_image_comap_of_finite_of_free
    /-
      case inst
      R : Type u_1
      inst✝ : CommRing R
      f g : Polynomial R
      hg : g.Monic
      ⊢ Module.Finite R (HasQuotient.Quotient (Polynomial R) (Ideal.span (Singleton. …
    -/
  · exact .of_basis (AdjoinRoot.powerBasis' hg).basis
    /-
      🎉 no goals
    -/
    /-
      case inst
      R : Type u_1
      inst✝ : CommRing R
      f g : Polynomial R
      hg : g.Monic
      ⊢ Module.Free R (HasQuotient.Quotient (Polynomial R) (Ideal.span (Singleton.si …
    -/
  · exact .of_basis (AdjoinRoot.powerBasis' hg).basis
    /-
      🎉 no goals
    -/


lemma isCompact_image_comap_of_monic (f g : R[X]) (hg : g.Monic) :
    IsCompact (comap C '' (zeroLocus {g} \ zeroLocus {f})) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    ⊢ IsCompact (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (Pri …
  -/
  obtain ⟨t, ht⟩ := exists_image_comap_of_monic f g hg
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    t : Finset R
    ht : Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeSp …
    ⊢ IsCompact (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (Pri …
  -/
  rw [ht, ← t.toSet.iUnion_of_singleton_coe, zeroLocus_iUnion, Set.compl_iInter]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    t : Finset R
    ht : Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeSp …
    ⊢ IsCompact (Set.iUnion fun i => HasCompl.compl (PrimeSpectrum.zeroLocus (Sing …
  -/
  apply isCompact_iUnion
  /-
    case intro.h
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    t : Finset R
    ht : Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeSp …
    ⊢ ∀ (i : ↑↑t), IsCompact (HasCompl.compl (PrimeSpectrum.zeroLocus (Singleton.s …
  -/
  exact fun _ ↦ by simpa using isCompact_basicOpen _
  /-
    🎉 no goals
  -/


lemma isOpen_image_comap_of_monic (f g : R[X]) (hg : g.Monic) :
    IsOpen (comap C '' (zeroLocus {g} \ zeroLocus {f})) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeS …
  -/
  obtain ⟨t, ht⟩ := exists_image_comap_of_monic f g hg
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    t : Finset R
    ht : Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeSp …
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeS …
  -/
  rw [ht]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    f g : Polynomial R
    hg : g.Monic
    t : Finset R
    ht : Eq (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (SDiff.sdiff (PrimeSp …
    ⊢ IsOpen (HasCompl.compl (PrimeSpectrum.zeroLocus ↑t))
  -/
  exact (isClosed_zeroLocus (R := R) t).isOpen_compl
  /-
    🎉 no goals
  -/


lemma mem_image_comap_C_basicOpen (f : MvPolynomial σ R) (x : PrimeSpectrum R) :
    x ∈ comap (C (σ := σ)) '' basicOpen f ↔ ∃ i, f.coeff i ∉ x.asIdeal := by
  classical
  trans f.map (algebraMap R x.asIdeal.ResidueField) ≠ 0
  · refine (mem_image_comap_basicOpen _ _).trans (not_iff_not.mpr ?_)
    let e : MvPolynomial σ R ⊗[R] x.asIdeal.ResidueField ≃ₐ[R]
        MvPolynomial σ x.asIdeal.ResidueField := scalarRTensorAlgEquiv
    rw [← IsNilpotent.map_iff e.injective, isNilpotent_iff_eq_zero]
    show (e.toAlgHom.toRingHom).comp (algebraMap _ _) f = 0 ↔ MvPolynomial.map _ f = 0
    congr!
    ext
    · simp [scalarRTensorAlgEquiv, e, coeff_map,
        Algebra.smul_def, apply_ite (f := algebraMap _ _)]
    · simp [e, monomial_one_one_eq_X, scalarRTensorAlgEquiv, coeff_map, coeff_X']
  · simp [MvPolynomial.ext_iff, coeff_map]


lemma image_comap_C_basicOpen (f : MvPolynomial σ R) :
      comap (C (σ := σ)) '' basicOpen f = (zeroLocus (Set.range f.coeff))ᶜ := by
    /-
      R : Type u_2
      inst✝ : CommRing R
      σ : Type u_1
      f : MvPolynomial σ R
      ⊢ Eq (Set.image ⇑(PrimeSpectrum.comap MvPolynomial.C) ↑(PrimeSpectrum.basicOpe …
    -/
    ext p
    /-
      case h
      R : Type u_2
      inst✝ : CommRing R
      σ : Type u_1
      f : MvPolynomial σ R
      p : PrimeSpectrum R
      ⊢ Iff (Membership.mem (Set.image ⇑(PrimeSpectrum.comap MvPolynomial.C) ↑(Prime …
    -/
    rw [mem_image_comap_C_basicOpen]
    /-
      case h
      R : Type u_2
      inst✝ : CommRing R
      σ : Type u_1
      f : MvPolynomial σ R
      p : PrimeSpectrum R
      ⊢ Iff (Exists fun i => Not (Membership.mem p.asIdeal (MvPolynomial.coeff i f)) …
    -/
    simp [Set.range_subset_iff]
    /-
      🎉 no goals
    -/


lemma isOpenMap_comap_C : IsOpenMap (comap (R := R) (C (σ := σ))) := by
  /-
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    ⊢ IsOpenMap ⇑(PrimeSpectrum.comap MvPolynomial.C)
  -/
  intro U hU
  /-
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    U : Set (PrimeSpectrum (MvPolynomial σ R))
    hU : IsOpen U
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap MvPolynomial.C)) U)
  -/
  obtain ⟨S, hS, rfl⟩ := isTopologicalBasis_basic_opens.open_eq_sUnion hU
  /-
    case intro.intro
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap MvPolynomial.C)) S.sUnion)
  -/
  rw [Set.image_sUnion]
  /-
    case intro.intro
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ IsOpen (Set.image (Set.image ⇑(PrimeSpectrum.comap MvPolynomial.C)) S).sUnion
  -/
  apply isOpen_sUnion
  /-
    case intro.intro.h
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    ⊢ ∀ (t : Set (PrimeSpectrum R)), Membership.mem (Set.image (Set.image ⇑(PrimeS …
  -/
  rintro _ ⟨t, ht, rfl⟩
  /-
    case intro.intro.h.intro.intro
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    t : Set (PrimeSpectrum (MvPolynomial σ R))
    ht : Membership.mem S t
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap MvPolynomial.C)) t)
  -/
  obtain ⟨r, rfl⟩ := hS ht
  /-
    case intro.intro.h.intro.intro.intro
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    r : MvPolynomial σ R
    ht : Membership.mem S ((fun r => ↑(PrimeSpectrum.basicOpen r)) r)
    ⊢ IsOpen (Set.image (⇑(PrimeSpectrum.comap MvPolynomial.C)) ((fun r => ↑(Prime …
  -/
  simp only [image_comap_C_basicOpen]
  /-
    case intro.intro.h.intro.intro.intro
    R : Type u_2
    inst✝ : CommRing R
    σ : Type u_1
    S : Set (Set (PrimeSpectrum (MvPolynomial σ R)))
    hS : HasSubset.Subset S (Set.range fun r => ↑(PrimeSpectrum.basicOpen r))
    hU : IsOpen S.sUnion
    r : MvPolynomial σ R
    ht : Membership.mem S ((fun r => ↑(PrimeSpectrum.basicOpen r)) r)
    ⊢ IsOpen (HasCompl.compl (PrimeSpectrum.zeroLocus (Set.range fun m => MvPolyno …
  -/
  exact (isClosed_zeroLocus _).isOpen_compl
  /-
    🎉 no goals
  -/


