/-- `charpoly f` is the characteristic polynomial of the matrix of `f` in any basis. -/
@[simp]
theorem charpoly_toMatrix {ι : Type w} [DecidableEq ι] [Fintype ι] (b : Basis ι R M) :
    (toMatrix b b f).charpoly = f.charpoly := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let A := toMatrix b b f
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let b' := chooseBasis R M
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let ι' := ChooseBasisIndex R M
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let A' := toMatrix b' b' f
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let e := Basis.indexEquiv b b'
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let φ := reindexLinearEquiv R R e e
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    φ : LinearEquiv (RingHom.id R) (Matrix ι ι R) (Matrix (Module.Free.ChooseBasis …
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let φ₁ := reindexLinearEquiv R R e (Equiv.refl ι')
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    φ : LinearEquiv (RingHom.id R) (Matrix ι ι R) (Matrix (Module.Free.ChooseBasis …
    φ₁ : LinearEquiv (RingHom.id R) (Matrix ι ι' R) (Matrix (Module.Free.ChooseBas …
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let φ₂ := reindexLinearEquiv R R (Equiv.refl ι') (Equiv.refl ι')
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    φ : LinearEquiv (RingHom.id R) (Matrix ι ι R) (Matrix (Module.Free.ChooseBasis …
    φ₁ : LinearEquiv (RingHom.id R) (Matrix ι ι' R) (Matrix (Module.Free.ChooseBas …
    φ₂ : LinearEquiv (RingHom.id R) (Matrix ι' ι' R) (Matrix ι' ι' R) := Matrix.re …
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let φ₃ := reindexLinearEquiv R R (Equiv.refl ι') e
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    φ : LinearEquiv (RingHom.id R) (Matrix ι ι R) (Matrix (Module.Free.ChooseBasis …
    φ₁ : LinearEquiv (RingHom.id R) (Matrix ι ι' R) (Matrix (Module.Free.ChooseBas …
    φ₂ : LinearEquiv (RingHom.id R) (Matrix ι' ι' R) (Matrix ι' ι' R) := Matrix.re …
    φ₃ : LinearEquiv (RingHom.id R) (Matrix ι' ι R) (Matrix ι' (Module.Free.Choose …
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let P := b.toMatrix b'
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Nontrivial R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ι : Type w
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    A : Matrix ι ι R := (LinearMap.toMatrix b b) f
    b' : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ι' : Type u_2 := Module.Free.ChooseBasisIndex R M
    A' : Matrix (Module.Free.ChooseBasisIndex R M) (Module.Free.ChooseBasisIndex R …
    e : Equiv ι (Module.Free.ChooseBasisIndex R M) := b.indexEquiv b'
    φ : LinearEquiv (RingHom.id R) (Matrix ι ι R) (Matrix (Module.Free.ChooseBasis …
    φ₁ : LinearEquiv (RingHom.id R) (Matrix ι ι' R) (Matrix (Module.Free.ChooseBas …
    φ₂ : LinearEquiv (RingHom.id R) (Matrix ι' ι' R) (Matrix ι' ι' R) := Matrix.re …
    φ₃ : LinearEquiv (RingHom.id R) (Matrix ι' ι R) (Matrix ι' (Module.Free.Choose …
    P : Matrix ι (Module.Free.ChooseBasisIndex R M) R := b.toMatrix ⇑b'
    ⊢ Eq ((LinearMap.toMatrix b b) f).charpoly f.charpoly
  -/
  let Q := b'.toMatrix b
  have hPQ : C.mapMatrix (φ₁ P) * C.mapMatrix (φ₃ Q) = 1 := by
    rw [RingHom.mapMatrix_apply, RingHom.mapMatrix_apply, ← Matrix.map_mul,
      reindexLinearEquiv_mul R R, Basis.toMatrix_mul_toMatrix_flip,
      reindexLinearEquiv_one, ← RingHom.mapMatrix_apply, RingHom.map_one]
  calc
    A.charpoly = (reindex e e A).charpoly := (charpoly_reindex _ _).symm
    _ = det (scalar ι' X - C.mapMatrix (φ A)) := rfl
    _ = det (scalar ι' X - C.mapMatrix (φ (P * A' * Q))) := by
      rw [basis_toMatrix_mul_linearMap_toMatrix_mul_basis_toMatrix]
    _ = det (scalar ι' X - C.mapMatrix (φ₁ P * φ₂ A' * φ₃ Q)) := by
      rw [reindexLinearEquiv_mul, reindexLinearEquiv_mul]
    _ = det (scalar ι' X - C.mapMatrix (φ₁ P) * C.mapMatrix A' * C.mapMatrix (φ₃ Q)) := by
      simp [φ₁, φ₂, φ₃, ι']
    _ = det (scalar ι' X * C.mapMatrix (φ₁ P) * C.mapMatrix (φ₃ Q) -
          C.mapMatrix (φ₁ P) * C.mapMatrix A' * C.mapMatrix (φ₃ Q)) := by
      rw [Matrix.mul_assoc ((scalar ι') X), hPQ, Matrix.mul_one]
    _ = det (C.mapMatrix (φ₁ P) * scalar ι' X * C.mapMatrix (φ₃ Q) -
          C.mapMatrix (φ₁ P) * C.mapMatrix A' * C.mapMatrix (φ₃ Q)) := by
      rw [scalar_commute _ commute_X]
    _ = det (C.mapMatrix (φ₁ P) * (scalar ι' X - C.mapMatrix A') * C.mapMatrix (φ₃ Q)) := by
      rw [← Matrix.sub_mul, ← Matrix.mul_sub]
    _ = det (C.mapMatrix (φ₁ P)) * det (scalar ι' X - C.mapMatrix A') * det (C.mapMatrix (φ₃ Q)) :=
      by rw [det_mul, det_mul]
    _ = det (C.mapMatrix (φ₁ P)) * det (C.mapMatrix (φ₃ Q)) * det (scalar ι' X - C.mapMatrix A') :=
      by ring
    _ = det (scalar ι' X - C.mapMatrix A') := by
      rw [← det_mul, hPQ, det_one, one_mul]
    _ = f.charpoly := rfl


lemma charpoly_prodMap (f₁ : M₁ →ₗ[R] M₁) (f₂ : M₂ →ₗ[R] M₂) :
    (f₁.prodMap f₂).charpoly = f₁.charpoly * f₂.charpoly := by
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    f₁ : LinearMap (RingHom.id R) M₁ M₁
    f₂ : LinearMap (RingHom.id R) M₂ M₂
    ⊢ Eq (f₁.prodMap f₂).charpoly (HMul.hMul f₁.charpoly f₂.charpoly)
  -/
  let b₁ := chooseBasis R M₁
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    f₁ : LinearMap (RingHom.id R) M₁ M₁
    f₂ : LinearMap (RingHom.id R) M₂ M₂
    b₁ : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis …
    ⊢ Eq (f₁.prodMap f₂).charpoly (HMul.hMul f₁.charpoly f₂.charpoly)
  -/
  let b₂ := chooseBasis R M₂
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    f₁ : LinearMap (RingHom.id R) M₁ M₁
    f₂ : LinearMap (RingHom.id R) M₂ M₂
    b₁ : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis …
    b₂ : Basis (Module.Free.ChooseBasisIndex R M₂) R M₂ := Module.Free.chooseBasis …
    ⊢ Eq (f₁.prodMap f₂).charpoly (HMul.hMul f₁.charpoly f₂.charpoly)
  -/
  let b := b₁.prod b₂
  rw [← charpoly_toMatrix f₁ b₁, ← charpoly_toMatrix f₂ b₂, ← charpoly_toMatrix (f₁.prodMap f₂) b,
    toMatrix_prodMap b₁ b₂ f₁ f₂, Matrix.charpoly_fromBlocks_zero₁₂]


@[simp]
lemma LinearEquiv.charpoly_conj (e : M₁ ≃ₗ[R] M₂) (φ : Module.End R M₁) :
    (e.conj φ).charpoly = φ.charpoly := by
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    e : LinearEquiv (RingHom.id R) M₁ M₂
    φ : Module.End R M₁
    ⊢ Eq (LinearMap.charpoly (e.conj φ)) (LinearMap.charpoly φ)
  -/
  let b := chooseBasis R M₁
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    e : LinearEquiv (RingHom.id R) M₁ M₂
    φ : Module.End R M₁
    b : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis  …
    ⊢ Eq (LinearMap.charpoly (e.conj φ)) (LinearMap.charpoly φ)
  -/
  rw [← LinearMap.charpoly_toMatrix φ b, ← LinearMap.charpoly_toMatrix (e.conj φ) (b.map e)]
  /-
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    e : LinearEquiv (RingHom.id R) M₁ M₂
    φ : Module.End R M₁
    b : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis  …
    ⊢ Eq ((LinearMap.toMatrix (b.map e) (b.map e)) (e.conj φ)).charpoly ((LinearMa …
  -/
  congr 1
  /-
    case e_M
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    e : LinearEquiv (RingHom.id R) M₁ M₂
    φ : Module.End R M₁
    b : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis  …
    ⊢ Eq ((LinearMap.toMatrix (b.map e) (b.map e)) (e.conj φ)) ((LinearMap.toMatri …
  -/
  ext i j : 1
  /-
    case e_M.a
    R : Type u_1
    M₁ : Type u_3
    M₂ : Type u_4
    inst✝⁹ : CommRing R
    inst✝⁸ : Nontrivial R
    inst✝⁷ : AddCommGroup M₁
    inst✝⁶ : Module R M₁
    inst✝⁵ : Module.Finite R M₁
    inst✝⁴ : Module.Free R M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : Module.Finite R M₂
    inst✝ : Module.Free R M₂
    e : LinearEquiv (RingHom.id R) M₁ M₂
    φ : Module.End R M₁
    b : Basis (Module.Free.ChooseBasisIndex R M₁) R M₁ := Module.Free.chooseBasis  …
    i j : Module.Free.ChooseBasisIndex R M₁
    ⊢ Eq ((LinearMap.toMatrix (b.map e) (b.map e)) (e.conj φ) i j) ((LinearMap.toM …
  -/
  simp [Matrix.charmatrix, LinearMap.toMatrix, Matrix.diagonal, LinearEquiv.conj_apply]
  /-
    🎉 no goals
  -/

