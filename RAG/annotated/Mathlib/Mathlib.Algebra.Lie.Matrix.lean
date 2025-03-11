/-- The natural equivalence between linear endomorphisms of finite free modules and square matrices
is compatible with the Lie algebra structures. -/
def lieEquivMatrix' : Module.End R (n → R) ≃ₗ⁅R⁆ Matrix n n R :=
  { LinearMap.toMatrix' with
    map_lie' := fun {T S} => by
      /-
        R : Type u
        inst✝² : CommRing R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        T S : Module.End R (n → R)
        ⊢ Eq ((↑__src✝).toFun (Bracket.bracket T S)) (Bracket.bracket ((↑__src✝).toFun …
      -/
      let f := @LinearMap.toMatrix' R _ n n _ _
      /-
        R : Type u
        inst✝² : CommRing R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        T S : Module.End R (n → R)
        f : LinearEquiv (RingHom.id R) (LinearMap (RingHom.id R) (n → R) (n → R)) (Mat …
        ⊢ Eq ((↑__src✝).toFun (Bracket.bracket T S)) (Bracket.bracket ((↑__src✝).toFun …
      -/
      change f (T.comp S - S.comp T) = f T * f S - f S * f T
      /-
        R : Type u
        inst✝² : CommRing R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        T S : Module.End R (n → R)
        f : LinearEquiv (RingHom.id R) (LinearMap (RingHom.id R) (n → R) (n → R)) (Mat …
        ⊢ Eq (f (HSub.hSub (LinearMap.comp T S) (LinearMap.comp S T))) (HSub.hSub (HMu …
      -/
      have h : ∀ T S : Module.End R _, f (T.comp S) = f T * f S := LinearMap.toMatrix'_comp
      /-
        R : Type u
        inst✝² : CommRing R
        n : Type w
        inst✝¹ : DecidableEq n
        inst✝ : Fintype n
        T S : Module.End R (n → R)
        f : LinearEquiv (RingHom.id R) (LinearMap (RingHom.id R) (n → R) (n → R)) (Mat …
        h : ∀ (T S : Module.End R (n → R)), Eq (f (LinearMap.comp T S)) (HMul.hMul (f  …
        ⊢ Eq (f (HSub.hSub (LinearMap.comp T S) (LinearMap.comp S T))) (HSub.hSub (HMu …
      -/
      rw [map_sub, h, h] }
      /-
        🎉 no goals
      -/


@[simp]
theorem lieEquivMatrix'_apply (f : Module.End R (n → R)) :
    lieEquivMatrix' f = LinearMap.toMatrix' f :=
  rfl


@[simp]
theorem lieEquivMatrix'_symm_apply (A : Matrix n n R) :
    (@lieEquivMatrix' R _ n _ _).symm A = Matrix.toLin' A :=
  rfl


/-- An invertible matrix induces a Lie algebra equivalence from the space of matrices to itself. -/
def Matrix.lieConj (P : Matrix n n R) (h : Invertible P) : Matrix n n R ≃ₗ⁅R⁆ Matrix n n R :=
  ((@lieEquivMatrix' R _ n _ _).symm.trans (P.toLinearEquiv' h).lieConj).trans lieEquivMatrix'


@[simp]
theorem Matrix.lieConj_apply (P A : Matrix n n R) (h : Invertible P) :
    P.lieConj h A = P * A * P⁻¹ := by
  simp [LinearEquiv.conj_apply, Matrix.lieConj, LinearMap.toMatrix'_comp,
    LinearMap.toMatrix'_toLin']


@[simp]
theorem Matrix.lieConj_symm_apply (P A : Matrix n n R) (h : Invertible P) :
    (P.lieConj h).symm A = P⁻¹ * A * P := by
  simp [LinearEquiv.symm_conj_apply, Matrix.lieConj, LinearMap.toMatrix'_comp,
    LinearMap.toMatrix'_toLin']


/-- For square matrices, the natural map that reindexes a matrix's rows and columns with equivalent
types, `Matrix.reindex`, is an equivalence of Lie algebras. -/
def Matrix.reindexLieEquiv : Matrix n n R ≃ₗ⁅R⁆ Matrix m m R :=
  { Matrix.reindexLinearEquiv R R e e with
    toFun := Matrix.reindex e e
    map_lie' := fun {_ _} => by
      simp only [LieRing.of_associative_ring_bracket, Matrix.reindex_apply,
        Matrix.submatrix_mul_equiv, Matrix.submatrix_sub, Pi.sub_apply] }


@[simp]
theorem Matrix.reindexLieEquiv_apply (M : Matrix n n R) :
    Matrix.reindexLieEquiv e M = Matrix.reindex e e M :=
  rfl


@[simp]
theorem Matrix.reindexLieEquiv_symm :
    (Matrix.reindexLieEquiv e : _ ≃ₗ⁅R⁆ _).symm = Matrix.reindexLieEquiv e.symm :=
  rfl


