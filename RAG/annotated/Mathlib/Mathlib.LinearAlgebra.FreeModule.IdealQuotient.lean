/-- We can write the quotient of an ideal over a PID as a product of quotients by principal ideals.
-/
noncomputable def quotientEquivPiSpan (I : Ideal S) (b : Basis ι R S) (hI : I ≠ ⊥) :
    (S ⧸ I) ≃ₗ[R] ∀ i, R ⧸ span ({I.smithCoeffs b hI i} : Set R) := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  haveI := Fintype.ofFinite ι
  -- Choose `e : S ≃ₗ I` and a basis `b'` for `S` that turns the map
  -- `f := ((Submodule.subtype I).restrictScalars R).comp e` into a diagonal matrix:
  -- there is an `a : ι → ℤ` such that `f (b' i) = a i • b' i`.
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    this : Fintype ι
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  let a := I.smithCoeffs b hI
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    this : Fintype ι
    a : ι → R := Ideal.smithCoeffs b I hI
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  let b' := I.ringBasis b hI
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    this : Fintype ι
    a : ι → R := Ideal.smithCoeffs b I hI
    b' : Basis ι R S := Ideal.ringBasis b I hI
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  let ab := I.selfBasis b hI
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    this : Fintype ι
    a : ι → R := Ideal.smithCoeffs b I hI
    b' : Basis ι R S := Ideal.ringBasis b I hI
    ab : Basis ι R (Subtype fun x => Membership.mem I x) := Ideal.selfBasis b I hI
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  have ab_eq := I.selfBasis_def b hI
  have mem_I_iff : ∀ x, x ∈ I ↔ ∀ i, a i ∣ b'.repr x i := by
    intro x
    simp_rw [ab.mem_ideal_iff', ab, ab_eq]
    have : ∀ (c : ι → R) (i), b'.repr (∑ j : ι, c j • a j • b' j) i = a i * c i := by
      intro c i
      simp only [← MulAction.mul_smul, b'.repr_sum_self, mul_comm]
    constructor
    · rintro ⟨c, rfl⟩ i
      exact ⟨c i, this c i⟩
    · rintro ha
      choose c hc using ha
      exact ⟨c, b'.ext_elem fun i => Eq.trans (hc i) (this c i).symm⟩
  -- Now we map everything through the linear equiv `S ≃ₗ (ι → R)`,
  -- which maps `I` to `I' := Π i, a i ℤ`.
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    inst✝³ : IsDomain R
    inst✝² : IsPrincipalIdealRing R
    inst✝¹ : IsDomain S
    inst✝ : Finite ι
    I : Ideal S
    b : Basis ι R S
    hI : Ne I Bot.bot
    this : Fintype ι
    a : ι → R := Ideal.smithCoeffs b I hI
    b' : Basis ι R S := Ideal.ringBasis b I hI
    ab : Basis ι R (Subtype fun x => Membership.mem I x) := Ideal.selfBasis b I hI
    ab_eq : ∀ (i : ι), Eq (↑((Ideal.selfBasis b I hI) i)) (HSMul.hSMul (Ideal.smit …
    mem_I_iff : ∀ (x : S), Iff (Membership.mem I x) (∀ (i : ι), Dvd.dvd (a i) ((b' …
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient S I) ((i : ι) → HasQuotient …
  -/
  let I' : Submodule R (ι → R) := Submodule.pi Set.univ fun i => span ({a i} : Set R)
  have : Submodule.map (b'.equivFun : S →ₗ[R] ι → R) (I.restrictScalars R) = I' := by
    ext x
    simp only [I', Submodule.mem_map, Submodule.mem_pi, mem_span_singleton, Set.mem_univ,
      Submodule.restrictScalars_mem, mem_I_iff, smul_eq_mul, forall_true_left, LinearEquiv.coe_coe,
      Basis.equivFun_apply]
    constructor
    · rintro ⟨y, hy, rfl⟩ i
      exact hy i
    · rintro hdvd
      refine ⟨∑ i, x i • b' i, fun i => ?_, ?_⟩ <;> rw [b'.repr_sum_self]
      · exact hdvd i
  refine ((Submodule.Quotient.restrictScalarsEquiv R I).restrictScalars R).symm.trans
    (σ₁₂ := RingHom.id R) (σ₃₂ := RingHom.id R) (re₂₃ := inferInstance) (re₃₂ := inferInstance) ?_
  refine (Submodule.Quotient.equiv (I.restrictScalars R) I' b'.equivFun this).trans
    (σ₁₂ := RingHom.id R) (σ₃₂ := RingHom.id R) (re₂₃ := inferInstance) (re₃₂ := inferInstance) ?_
  classical
    let this :=
      Submodule.quotientPi (show _ → Submodule R R from fun i => span ({a i} : Set R))
    exact this


/-- Ideal quotients over a free finite extension of `ℤ` are isomorphic to a direct product of
`ZMod`. -/
noncomputable def quotientEquivPiZMod (I : Ideal S) (b : Basis ι ℤ S) (hI : I ≠ ⊥) :
    S ⧸ I ≃+ ∀ i, ZMod (I.smithCoeffs b hI i).natAbs :=
  let a := I.smithCoeffs b hI
  let e := I.quotientEquivPiSpan b hI
  let e' : (∀ i : ι, ℤ ⧸ span ({a i} : Set ℤ)) ≃+ ∀ i : ι, ZMod (a i).natAbs :=
    AddEquiv.piCongrRight fun i => ↑(Int.quotientSpanEquivZMod (a i))
  (↑(e : (S ⧸ I) ≃ₗ[ℤ] _) : S ⧸ I ≃+ _).trans e'


/-- A nonzero ideal over a free finite extension of `ℤ` has a finite quotient.

Can't be an instance because of the side condition `I ≠ ⊥`, and more importantly,
because the choice of `Fintype` instance is non-canonical.
-/
noncomputable def fintypeQuotientOfFreeOfNeBot [Module.Free ℤ S] [Module.Finite ℤ S]
    (I : Ideal S) (hI : I ≠ ⊥) : Fintype (S ⧸ I) := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    inst✝³ : IsDomain S
    inst✝² : Finite ι
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    hI : Ne I Bot.bot
    ⊢ Fintype (HasQuotient.Quotient S I)
  -/
  let b := Module.Free.chooseBasis ℤ S
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    inst✝³ : IsDomain S
    inst✝² : Finite ι
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    hI : Ne I Bot.bot
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Fintype (HasQuotient.Quotient S I)
  -/
  let a := I.smithCoeffs b hI
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    inst✝³ : IsDomain S
    inst✝² : Finite ι
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    hI : Ne I Bot.bot
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    ⊢ Fintype (HasQuotient.Quotient S I)
  -/
  let e := I.quotientEquivPiZMod b hI
  haveI : ∀ i, NeZero (a i).natAbs := fun i =>
    ⟨Int.natAbs_ne_zero.mpr (smithCoeffs_ne_zero b I hI i)⟩
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    inst✝⁵ : IsDomain R
    inst✝⁴ : IsPrincipalIdealRing R
    inst✝³ : IsDomain S
    inst✝² : Finite ι
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    hI : Ne I Bot.bot
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    e : AddEquiv (HasQuotient.Quotient S I) ((i : Module.Free.ChooseBasisIndex Int …
    this : ∀ (i : Module.Free.ChooseBasisIndex Int S), NeZero (a i).natAbs
    ⊢ Fintype (HasQuotient.Quotient S I)
  -/
  classical exact Fintype.ofEquiv (∀ i, ZMod (a i).natAbs) e.symm
  /-
    🎉 no goals
  -/


/-- Decompose `S⧸I` as a direct sum of cyclic `R`-modules
  (quotients by the ideals generated by Smith coefficients of `I`). -/
noncomputable def quotientEquivDirectSum :
    (S ⧸ I) ≃ₗ[F] ⨁ i, R ⧸ span ({I.smithCoeffs b hI i} : Set R) := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    inst✝⁷ : IsDomain R
    inst✝⁶ : IsPrincipalIdealRing R
    inst✝⁵ : IsDomain S
    inst✝⁴ : Finite ι
    F : Type u_4
    inst✝³ : CommRing F
    inst✝² : Algebra F R
    inst✝¹ : Algebra F S
    inst✝ : IsScalarTower F R S
    b : Basis ι R S
    I : Ideal S
    hI : Ne I Bot.bot
    ⊢ LinearEquiv (RingHom.id F) (HasQuotient.Quotient S I) (DirectSum ι fun i =>  …
  -/
  haveI := Fintype.ofFinite ι
  -- Porting note: manual construction of `CompatibleSMul` typeclass no longer needed
  exact ((I.quotientEquivPiSpan b _).restrictScalars F).trans
    (DirectSum.linearEquivFunOnFintype _ _ _).symm


theorem finrank_quotient_eq_sum {ι} [Fintype ι] (b : Basis ι R S) [Nontrivial F]
    [∀ i, Module.Free F (R ⧸ span ({I.smithCoeffs b hI i} : Set R))]
    [∀ i, Module.Finite F (R ⧸ span ({I.smithCoeffs b hI i} : Set R))] :
    Module.finrank F (S ⧸ I) =
      ∑ i, Module.finrank F (R ⧸ span ({I.smithCoeffs b hI i} : Set R)) := by
  -- slow, and dot notation doesn't work
  /-
    R : Type u_2
    S : Type u_3
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : IsDomain R
    inst✝⁹ : IsPrincipalIdealRing R
    inst✝⁸ : IsDomain S
    F : Type u_4
    inst✝⁷ : CommRing F
    inst✝⁶ : Algebra F R
    inst✝⁵ : Algebra F S
    inst✝⁴ : IsScalarTower F R S
    I : Ideal S
    hI : Ne I Bot.bot
    ι : Type u_5
    inst✝³ : Fintype ι
    b : Basis ι R S
    inst✝² : Nontrivial F
    inst✝¹ : ∀ (i : ι), Module.Free F (HasQuotient.Quotient R (Ideal.span (Singlet …
    inst✝ : ∀ (i : ι), Module.Finite F (HasQuotient.Quotient R (Ideal.span (Single …
    ⊢ Eq (Module.finrank F (HasQuotient.Quotient S I)) (Finset.univ.sum fun i => M …
  -/
  rw [LinearEquiv.finrank_eq <| quotientEquivDirectSum F b hI, Module.finrank_directSum]
  /-
    🎉 no goals
  -/


