/-- The composition of a matrix (as an endomorphism of `ι → R`) with the projection
`(ι → R) →ₗ[R] M`. -/
def PiToModule.fromMatrix [DecidableEq ι] : Matrix ι ι R →ₗ[R] (ι → R) →ₗ[R] M :=
  (LinearMap.llcomp R _ _ _ (Fintype.linearCombination R R b)).comp algEquivMatrix'.symm.toLinearMap


theorem PiToModule.fromMatrix_apply [DecidableEq ι] (A : Matrix ι ι R) (w : ι → R) :
    PiToModule.fromMatrix R b A w = Fintype.linearCombination R R b (A *ᵥ w) :=
  rfl


theorem PiToModule.fromMatrix_apply_single_one [DecidableEq ι] (A : Matrix ι ι R) (j : ι) :
    PiToModule.fromMatrix R b A (Pi.single j 1) = ∑ i : ι, A i j • b i := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A : Matrix ι ι R
    j : ι
    ⊢ Eq (((PiToModule.fromMatrix R b) A) (Pi.single j 1)) (Finset.univ.sum fun i  …
  -/
  rw [PiToModule.fromMatrix_apply, Fintype.linearCombination_apply, Matrix.mulVec_single]
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A : Matrix ι ι R
    j : ι
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((fun i => HMul.hMul (A i j) 1) i)  …
  -/
  simp_rw [mul_one]
  /-
    🎉 no goals
  -/


/-- The endomorphisms of `M` acts on `(ι → R) →ₗ[R] M`, and takes the projection
to a `(ι → R) →ₗ[R] M`. -/
def PiToModule.fromEnd : Module.End R M →ₗ[R] (ι → R) →ₗ[R] M :=
  LinearMap.lcomp _ _ (Fintype.linearCombination R R b)


theorem PiToModule.fromEnd_apply (f : Module.End R M) (w : ι → R) :
    PiToModule.fromEnd R b f w = f (Fintype.linearCombination R R b w) :=
  rfl


theorem PiToModule.fromEnd_apply_single_one [DecidableEq ι] (f : Module.End R M) (i : ι) :
    PiToModule.fromEnd R b f (Pi.single i 1) = f (b i) := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    f : Module.End R M
    i : ι
    ⊢ Eq (((PiToModule.fromEnd R b) f) (Pi.single i 1)) (f (b i))
  -/
  rw [PiToModule.fromEnd_apply]
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    f : Module.End R M
    i : ι
    ⊢ Eq (f (((Fintype.linearCombination R R) b) (Pi.single i 1))) (f (b i))
  -/
  congr
  /-
    case h.e_6.h
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    f : Module.End R M
    i : ι
    ⊢ Eq (((Fintype.linearCombination R R) b) (Pi.single i 1)) (b i)
  -/
  convert Fintype.linearCombination_apply_single (S := R) R b i (1 : R)
  /-
    case h.e'_3
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    f : Module.End R M
    i : ι
    ⊢ Eq (b i) (HSMul.hSMul 1 (b i))
  -/
  rw [one_smul]
  /-
    🎉 no goals
  -/


theorem PiToModule.fromEnd_injective (hb : Submodule.span R (Set.range b) = ⊤) :
    Function.Injective (PiToModule.fromEnd R b) := by
  /-
    ι : Type u_1
    inst✝³ : Fintype ι
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    b : ι → M
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    ⊢ Function.Injective ⇑(PiToModule.fromEnd R b)
  -/
  intro x y e
  /-
    ι : Type u_1
    inst✝³ : Fintype ι
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    b : ι → M
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    x y : Module.End R M
    e : Eq ((PiToModule.fromEnd R b) x) ((PiToModule.fromEnd R b) y)
    ⊢ Eq x y
  -/
  ext m
  obtain ⟨m, rfl⟩ : m ∈ LinearMap.range (Fintype.linearCombination R R b) := by
    rw [(Fintype.range_linearCombination R b).trans hb]
    exact Submodule.mem_top
  /-
    case h.intro
    ι : Type u_1
    inst✝³ : Fintype ι
    M : Type u_2
    inst✝² : AddCommGroup M
    R : Type u_3
    inst✝¹ : CommRing R
    inst✝ : Module R M
    b : ι → M
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    x y : Module.End R M
    e : Eq ((PiToModule.fromEnd R b) x) ((PiToModule.fromEnd R b) y)
    m : ι → R
    ⊢ Eq (x (((Fintype.linearCombination R R) b) m)) (y (((Fintype.linearCombinati …
  -/
  exact (LinearMap.congr_fun e m : _)
  /-
    🎉 no goals
  -/


/-- We say that a matrix represents an endomorphism of `M` if the matrix acting on `ι → R` is
equal to `f` via the projection `(ι → R) →ₗ[R] M` given by a fixed (spanning) set. -/
def Matrix.Represents (A : Matrix ι ι R) (f : Module.End R M) : Prop :=
  PiToModule.fromMatrix R b A = PiToModule.fromEnd R b f


theorem Matrix.Represents.congr_fun {A : Matrix ι ι R} {f : Module.End R M} (h : A.Represents b f)
    (x) : Fintype.linearCombination R R b (A *ᵥ x) = f (Fintype.linearCombination R R b x) :=
  LinearMap.congr_fun h x


theorem Matrix.represents_iff {A : Matrix ι ι R} {f : Module.End R M} :
    A.Represents b f ↔
      ∀ x, Fintype.linearCombination R R b (A *ᵥ x) = f (Fintype.linearCombination R R b x) :=
  ⟨fun e x => e.congr_fun x, fun H => LinearMap.ext fun x => H x⟩


theorem Matrix.represents_iff' {A : Matrix ι ι R} {f : Module.End R M} :
    A.Represents b f ↔ ∀ j, ∑ i : ι, A i j • b i = f (b j) := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A : Matrix ι ι R
    f : Module.End R M
    ⊢ Iff (Matrix.Represents b A f) (∀ (j : ι), Eq (Finset.univ.sum fun i => HSMul …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      ⊢ Matrix.Represents b A f → ∀ (j : ι), Eq (Finset.univ.sum fun i => HSMul.hSMu …
    -/
  · intro h i
    /-
      case mp
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      h : Matrix.Represents b A f
      i : ι
      ⊢ Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A i_1 i) (b i_1)) (f (b i))
    -/
    have := LinearMap.congr_fun h (Pi.single i 1)
    /-
      case mp
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      h : Matrix.Represents b A f
      i : ι
      this : Eq (((PiToModule.fromMatrix R b) A) (Pi.single i 1)) (((PiToModule.from …
      ⊢ Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A i_1 i) (b i_1)) (f (b i))
    -/
    rwa [PiToModule.fromEnd_apply_single_one, PiToModule.fromMatrix_apply_single_one] at this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      ⊢ (∀ (j : ι), Eq (Finset.univ.sum fun i => HSMul.hSMul (A i j) (b i)) (f (b j) …
    -/
  · intro h
    /-
      case mpr
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      h : ∀ (j : ι), Eq (Finset.univ.sum fun i => HSMul.hSMul (A i j) (b i)) (f (b j))
      ⊢ Matrix.Represents b A f
    -/
    ext
    simp_rw [LinearMap.comp_apply, LinearMap.coe_single, PiToModule.fromEnd_apply_single_one,
      PiToModule.fromMatrix_apply_single_one]
    /-
      case mpr.h.h
      ι : Type u_1
      inst✝⁴ : Fintype ι
      M : Type u_2
      inst✝³ : AddCommGroup M
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Module R M
      b : ι → M
      inst✝ : DecidableEq ι
      A : Matrix ι ι R
      f : Module.End R M
      h : ∀ (j : ι), Eq (Finset.univ.sum fun i => HSMul.hSMul (A i j) (b i)) (f (b j))
      i✝ : ι
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (A i i✝) (b i)) (f (b i✝))
    -/
    apply h
    /-
      🎉 no goals
    -/


theorem Matrix.Represents.mul {A A' : Matrix ι ι R} {f f' : Module.End R M} (h : A.Represents b f)
    (h' : Matrix.Represents b A' f') : (A * A').Represents b (f * f') := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    ⊢ Matrix.Represents b (HMul.hMul A A') (HMul.hMul f f')
  -/
  delta Matrix.Represents PiToModule.fromMatrix
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    ⊢ Eq ((((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R R …
  -/
  rw [LinearMap.comp_apply, AlgEquiv.toLinearMap_apply, _root_.map_mul]
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    ⊢ Eq (((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R R) …
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    i✝ : ι
    ⊢ Eq (((((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R  …
  -/
  dsimp [PiToModule.fromEnd]
  /-
    case h.h
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    i✝ : ι
    ⊢ Eq (((Fintype.linearCombination R R) b) ((algEquivMatrix'.symm A) ((algEquiv …
  -/
  rw [← h'.congr_fun, ← h.congr_fun]
  /-
    case h.h
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    i✝ : ι
    ⊢ Eq (((Fintype.linearCombination R R) b) ((algEquivMatrix'.symm A) ((algEquiv …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Matrix.Represents.one : (1 : Matrix ι ι R).Represents b 1 := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    ⊢ Matrix.Represents b 1 1
  -/
  delta Matrix.Represents PiToModule.fromMatrix
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    ⊢ Eq ((((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R R …
  -/
  rw [LinearMap.comp_apply, AlgEquiv.toLinearMap_apply, _root_.map_one]
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    ⊢ Eq (((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R R) …
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    i✝ : ι
    ⊢ Eq (((((LinearMap.llcomp R (ι → R) (ι → R) M) ((Fintype.linearCombination R  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Matrix.Represents.add {A A' : Matrix ι ι R} {f f' : Module.End R M} (h : A.Represents b f)
    (h' : Matrix.Represents b A' f') : (A + A').Represents b (f + f') := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A A' : Matrix ι ι R
    f f' : Module.End R M
    h : Matrix.Represents b A f
    h' : Matrix.Represents b A' f'
    ⊢ Matrix.Represents b (HAdd.hAdd A A') (HAdd.hAdd f f')
  -/
  delta Matrix.Represents at h h' ⊢; rw [map_add, map_add, h, h']
                                     /-
                                       🎉 no goals
                                     -/


theorem Matrix.Represents.zero : (0 : Matrix ι ι R).Represents b 0 := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    ⊢ Matrix.Represents b 0 0
  -/
  delta Matrix.Represents
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    ⊢ Eq ((PiToModule.fromMatrix R b) 0) ((PiToModule.fromEnd R b) 0)
  -/
  rw [map_zero, map_zero]
  /-
    🎉 no goals
  -/


theorem Matrix.Represents.smul {A : Matrix ι ι R} {f : Module.End R M} (h : A.Represents b f)
    (r : R) : (r • A).Represents b (r • f) := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A : Matrix ι ι R
    f : Module.End R M
    h : Matrix.Represents b A f
    r : R
    ⊢ Matrix.Represents b (HSMul.hSMul r A) (HSMul.hSMul r f)
  -/
  delta Matrix.Represents at h ⊢
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    A : Matrix ι ι R
    f : Module.End R M
    h : Eq ((PiToModule.fromMatrix R b) A) ((PiToModule.fromEnd R b) f)
    r : R
    ⊢ Eq ((PiToModule.fromMatrix R b) (HSMul.hSMul r A)) ((PiToModule.fromEnd R b) …
  -/
  rw [_root_.map_smul, _root_.map_smul, h]
  /-
    🎉 no goals
  -/


theorem Matrix.Represents.algebraMap (r : R) :
    (algebraMap _ (Matrix ι ι R) r).Represents b (algebraMap _ (Module.End R M) r) := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    r : R
    ⊢ Matrix.Represents b ((_root_.algebraMap R (Matrix ι ι R)) r) ((_root_.algebr …
  -/
  simpa only [Algebra.algebraMap_eq_smul_one] using Matrix.Represents.one.smul r
  /-
    🎉 no goals
  -/


theorem Matrix.Represents.eq (hb : Submodule.span R (Set.range b) = ⊤)
    {A : Matrix ι ι R} {f f' : Module.End R M} (h : A.Represents b f)
    (h' : A.Represents b f') : f = f' :=
  PiToModule.fromEnd_injective R b hb (h.symm.trans h')


/-- The subalgebra of `Matrix ι ι R` that consists of matrices that actually represent
endomorphisms on `M`. -/
def Matrix.isRepresentation : Subalgebra R (Matrix ι ι R) where
  carrier := { A | ∃ f : Module.End R M, A.Represents b f }
  mul_mem' := fun ⟨f₁, e₁⟩ ⟨f₂, e₂⟩ => ⟨f₁ * f₂, e₁.mul e₂⟩
  one_mem' := ⟨1, Matrix.Represents.one⟩
  add_mem' := fun ⟨f₁, e₁⟩ ⟨f₂, e₂⟩ => ⟨f₁ + f₂, e₁.add e₂⟩
  zero_mem' := ⟨0, Matrix.Represents.zero⟩
  algebraMap_mem' r := ⟨algebraMap _ _ r, .algebraMap _⟩


/-- The map sending a matrix to the endomorphism it represents. This is an `R`-algebra morphism. -/
noncomputable def Matrix.isRepresentation.toEnd :
    Matrix.isRepresentation R b →ₐ[R] Module.End R M where
  toFun A := A.2.choose
  map_one' := (1 : Matrix.isRepresentation R b).2.choose_spec.eq hb Matrix.Represents.one
  map_mul' A₁ A₂ := (A₁ * A₂).2.choose_spec.eq hb (A₁.2.choose_spec.mul A₂.2.choose_spec)
  map_zero' := (0 : Matrix.isRepresentation R b).2.choose_spec.eq hb Matrix.Represents.zero
  map_add' A₁ A₂ := (A₁ + A₂).2.choose_spec.eq hb (A₁.2.choose_spec.add A₂.2.choose_spec)
  commutes' r :=
    (algebraMap _ (Matrix.isRepresentation R b) r).2.choose_spec.eq hb (.algebraMap r)


theorem Matrix.isRepresentation.toEnd_represents (A : Matrix.isRepresentation R b) :
    (A : Matrix ι ι R).Represents b (Matrix.isRepresentation.toEnd R b hb A) :=
  A.2.choose_spec


theorem Matrix.isRepresentation.eq_toEnd_of_represents (A : Matrix.isRepresentation R b)
    {f : Module.End R M} (h : (A : Matrix ι ι R).Represents b f) :
    Matrix.isRepresentation.toEnd R b hb A = f :=
  A.2.choose_spec.eq hb h


theorem Matrix.isRepresentation.toEnd_exists_mem_ideal (f : Module.End R M) (I : Ideal R)
    (hI : LinearMap.range f ≤ I • ⊤) :
    ∃ M, Matrix.isRepresentation.toEnd R b hb M = f ∧ ∀ i j, M.1 i j ∈ I := by
  have : ∀ x, f x ∈ LinearMap.range (Ideal.finsuppTotal ι M I b) := by
    rw [Ideal.range_finsuppTotal, hb]
    exact fun x => hI (LinearMap.mem_range_self f x)
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    f : Module.End R M
    I : Ideal R
    hI : LE.le (LinearMap.range f) (HSMul.hSMul I Top.top)
    this : ∀ (x : M), Membership.mem (LinearMap.range (Ideal.finsuppTotal ι M I b) …
    ⊢ Exists fun M_1 => And (Eq ((Matrix.isRepresentation.toEnd R b hb) M_1) f) (∀ …
  -/
  choose bM' hbM' using this
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    f : Module.End R M
    I : Ideal R
    hI : LE.le (LinearMap.range f) (HSMul.hSMul I Top.top)
    bM' : M → Finsupp ι (Subtype fun x => Membership.mem I x)
    hbM' : ∀ (x : M), Eq ((Ideal.finsuppTotal ι M I b) (bM' x)) (f x)
    ⊢ Exists fun M_1 => And (Eq ((Matrix.isRepresentation.toEnd R b hb) M_1) f) (∀ …
  -/
  let A : Matrix ι ι R := fun i j => bM' (b j) i
  have : A.Represents b f := by
    rw [Matrix.represents_iff']
    dsimp [A]
    intro j
    specialize hbM' (b j)
    rwa [Ideal.finsuppTotal_apply_eq_of_fintype] at hbM'
  exact
    ⟨⟨A, f, this⟩, Matrix.isRepresentation.eq_toEnd_of_represents R b hb ⟨A, f, this⟩ this,
      fun i j => (bM' (b j) i).prop⟩


theorem Matrix.isRepresentation.toEnd_surjective :
    Function.Surjective (Matrix.isRepresentation.toEnd R b hb) := by
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    ⊢ Function.Surjective ⇑(Matrix.isRepresentation.toEnd R b hb)
  -/
  intro f
  /-
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M : Type u_2
    inst✝³ : AddCommGroup M
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M
    b : ι → M
    inst✝ : DecidableEq ι
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    f : Module.End R M
    ⊢ Exists fun a => Eq ((Matrix.isRepresentation.toEnd R b hb) a) f
  -/
  obtain ⟨M, e, -⟩ := Matrix.isRepresentation.toEnd_exists_mem_ideal R b hb f ⊤ (by simp)
  /-
    case intro.intro
    ι : Type u_1
    inst✝⁴ : Fintype ι
    M✝ : Type u_2
    inst✝³ : AddCommGroup M✝
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Module R M✝
    b : ι → M✝
    inst✝ : DecidableEq ι
    hb : Eq (Submodule.span R (Set.range b)) Top.top
    f : Module.End R M✝
    M : Subtype fun x => Membership.mem (Matrix.isRepresentation R b) x
    e : Eq ((Matrix.isRepresentation.toEnd R b hb) M) f
    ⊢ Exists fun a => Eq ((Matrix.isRepresentation.toEnd R b hb) a) f
  -/
  exact ⟨M, e⟩
  /-
    🎉 no goals
  -/


/-- The **Cayley-Hamilton Theorem** for f.g. modules over arbitrary rings states that for each
`R`-endomorphism `φ` of an `R`-module `M` such that `φ(M) ≤ I • M` for some ideal `I`, there
exists some `n` and some `aᵢ ∈ Iⁱ` such that `φⁿ + a₁ φⁿ⁻¹ + ⋯ + aₙ = 0`.

This is the version found in Eisenbud 4.3, which is slightly weaker than Matsumura 2.1
(this lacks the constraint on `n`), and is slightly stronger than Atiyah-Macdonald 2.4.
-/
theorem LinearMap.exists_monic_and_coeff_mem_pow_and_aeval_eq_zero_of_range_le_smul
    [Module.Finite R M] (f : Module.End R M) (I : Ideal R) (hI : LinearMap.range f ≤ I • ⊤) :
    ∃ p : R[X], p.Monic ∧ (∀ k, p.coeff k ∈ I ^ (p.natDegree - k)) ∧ Polynomial.aeval f p = 0 := by
  classical
    cases subsingleton_or_nontrivial R
    · exact ⟨0, Polynomial.monic_of_subsingleton _, by simp⟩
    obtain ⟨s : Finset M, hs : Submodule.span R (s : Set M) = ⊤⟩ :=
      Module.Finite.out (R := R) (M := M)
    -- Porting note: `H` was `rfl`
    obtain ⟨A, H, h⟩ :=
      Matrix.isRepresentation.toEnd_exists_mem_ideal R ((↑) : s → M)
        (by rw [Subtype.range_coe_subtype, Finset.setOf_mem, hs]) f I hI
    rw [← H]
    refine ⟨A.1.charpoly, A.1.charpoly_monic, ?_, ?_⟩
    · rw [A.1.charpoly_natDegree_eq_dim]
      exact coeff_charpoly_mem_ideal_pow h
    · rw [Polynomial.aeval_algHom_apply,
        ← map_zero (Matrix.isRepresentation.toEnd R ((↑) : s → M) _)]
      congr 1
      ext1
      rw [Polynomial.aeval_subalgebra_coe, Matrix.aeval_self_charpoly, Subalgebra.coe_zero]


theorem LinearMap.exists_monic_and_aeval_eq_zero [Module.Finite R M] (f : Module.End R M) :
    ∃ p : R[X], p.Monic ∧ Polynomial.aeval f p = 0 :=
                                                                                         /-
                                                                                           M : Type u_2
                                                                                           inst✝³ : AddCommGroup M
                                                                                           R : Type u_3
                                                                                           inst✝² : CommRing R
                                                                                           inst✝¹ : Module R M
                                                                                           inst✝ : Module.Finite R M
                                                                                           f : Module.End R M
                                                                                           ⊢ LE.le (LinearMap.range f) (HSMul.hSMul Top.top Top.top)
                                                                                         -/
  (LinearMap.exists_monic_and_coeff_mem_pow_and_aeval_eq_zero_of_range_le_smul R f ⊤ (by simp)).imp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    fun _ h => h.imp_right And.right

