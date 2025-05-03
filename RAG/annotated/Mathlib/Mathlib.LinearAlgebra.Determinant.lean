/-- If `R^m` and `R^n` are linearly equivalent, then `m` and `n` are also equivalent. -/
def equivOfPiLEquivPi {R : Type*} [Finite m] [Finite n] [CommRing R] [Nontrivial R]
    (e : (m → R) ≃ₗ[R] n → R) : m ≃ n :=
  Basis.indexEquiv (Basis.ofEquivFun e.symm) (Pi.basisFun _ _)


/-- If `M` and `M'` are each other's inverse matrices, they are square matrices up to
equivalence of types. -/
def indexEquivOfInv [Nontrivial A] [DecidableEq m] [DecidableEq n] {M : Matrix m n A}
    {M' : Matrix n m A} (hMM' : M * M' = 1) (hM'M : M' * M = 1) : m ≃ n :=
  equivOfPiLEquivPi (toLin'OfInv hMM' hM'M)


theorem det_comm [DecidableEq n] (M N : Matrix n n A) : det (M * N) = det (N * M) := by
  /-
    A : Type u_5
    inst✝² : CommRing A
    n : Type u_7
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M N : Matrix n n A
    ⊢ Eq (HMul.hMul M N).det (HMul.hMul N M).det
  -/
  rw [det_mul, det_mul, mul_comm]
  /-
    🎉 no goals
  -/


/-- If there exists a two-sided inverse `M'` for `M` (indexed differently),
then `det (N * M) = det (M * N)`. -/
theorem det_comm' [DecidableEq m] [DecidableEq n] {M : Matrix n m A} {N : Matrix m n A}
    {M' : Matrix m n A} (hMM' : M * M' = 1) (hM'M : M' * M = 1) : det (M * N) = det (N * M) := by
  /-
    A : Type u_5
    inst✝⁴ : CommRing A
    m : Type u_6
    n : Type u_7
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    M : Matrix n m A
    N M' : Matrix m n A
    hMM' : Eq (HMul.hMul M M') 1
    hM'M : Eq (HMul.hMul M' M) 1
    ⊢ Eq (HMul.hMul M N).det (HMul.hMul N M).det
  -/
  nontriviality A
  -- Although `m` and `n` are different a priori, we will show they have the same cardinality.
  -- This turns the problem into one for square matrices, which is easy.
  /-
    A : Type u_5
    inst✝⁴ : CommRing A
    m : Type u_6
    n : Type u_7
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    M : Matrix n m A
    N M' : Matrix m n A
    hMM' : Eq (HMul.hMul M M') 1
    hM'M : Eq (HMul.hMul M' M) 1
    a✝ : Nontrivial A
    ⊢ Eq (HMul.hMul M N).det (HMul.hMul N M).det
  -/
  let e := indexEquivOfInv hMM' hM'M
  rw [← det_submatrix_equiv_self e, ← submatrix_mul_equiv _ _ _ (Equiv.refl n) _, det_comm,
    submatrix_mul_equiv, Equiv.coe_refl, submatrix_id_id]


/-- If `M'` is a two-sided inverse for `M` (indexed differently), `det (M * N * M') = det N`.

See `Matrix.det_conj` and `Matrix.det_conj'` for the case when `M' = M⁻¹` or vice versa. -/
theorem det_conj_of_mul_eq_one [DecidableEq m] [DecidableEq n] {M : Matrix m n A}
    {M' : Matrix n m A} {N : Matrix n n A} (hMM' : M * M' = 1) (hM'M : M' * M = 1) :
    det (M * N * M') = det N := by
  /-
    A : Type u_5
    inst✝⁴ : CommRing A
    m : Type u_6
    n : Type u_7
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    M : Matrix m n A
    M' : Matrix n m A
    N : Matrix n n A
    hMM' : Eq (HMul.hMul M M') 1
    hM'M : Eq (HMul.hMul M' M) 1
    ⊢ Eq (HMul.hMul (HMul.hMul M N) M').det N.det
  -/
  rw [← det_comm' hM'M hMM', ← Matrix.mul_assoc, hM'M, Matrix.one_mul]
  /-
    🎉 no goals
  -/


/-- The determinant of `LinearMap.toMatrix` does not depend on the choice of basis. -/
theorem det_toMatrix_eq_det_toMatrix [DecidableEq κ] (b : Basis ι A M) (c : Basis κ A M)
    (f : M →ₗ[A] M) : det (LinearMap.toMatrix b b f) = det (LinearMap.toMatrix c c f) := by
  rw [← linearMap_toMatrix_mul_basis_toMatrix c b c, ← basis_toMatrix_mul_linearMap_toMatrix b c b,
      Matrix.det_conj_of_mul_eq_one] <;>
    /-
      case hMM'
      M : Type u_2
      inst✝⁶ : AddCommGroup M
      ι : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Fintype ι
      A : Type u_5
      inst✝³ : CommRing A
      inst✝² : Module A M
      κ : Type u_6
      inst✝¹ : Fintype κ
      inst✝ : DecidableEq κ
      b : Basis ι A M
      c : Basis κ A M
      f : LinearMap (RingHom.id A) M M
      ⊢ Eq (HMul.hMul (c.toMatrix ⇑b) (b.toMatrix ⇑c)) 1
    -/
    /-
      🎉 no goals
    -/
    rw [Basis.toMatrix_mul_toMatrix, Basis.toMatrix_self]
    /-
      🎉 no goals
    -/



/-- The determinant of an endomorphism given a basis.

See `LinearMap.det` for a version that populates the basis non-computably.

Although the `Trunc (Basis ι A M)` parameter makes it slightly more convenient to switch bases,
there is no good way to generalize over universe parameters, so we can't fully state in `detAux`'s
type that it does not depend on the choice of basis. Instead you can use the `detAux_def''` lemma,
or avoid mentioning a basis at all using `LinearMap.det`.
-/
irreducible_def detAux : Trunc (Basis ι A M) → (M →ₗ[A] M) →* A :=
  Trunc.lift
    (fun b : Basis ι A M => detMonoidHom.comp (toMatrixAlgEquiv b : (M →ₗ[A] M) →* Matrix ι ι A))
    fun b c => MonoidHom.ext <| det_toMatrix_eq_det_toMatrix b c


/-- Unfold lemma for `detAux`.

See also `detAux_def''` which allows you to vary the basis.
-/
theorem detAux_def' (b : Basis ι A M) (f : M →ₗ[A] M) :
    LinearMap.detAux (Trunc.mk b) f = Matrix.det (LinearMap.toMatrix b b f) := by
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b : Basis ι A M
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq ((LinearMap.detAux (Trunc.mk b)) f) ((LinearMap.toMatrix b b) f).det
  -/
  rw [detAux]
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b : Basis ι A M
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq ((Trunc.lift (fun b => Matrix.detMonoidHom.comp ↑(LinearMap.toMatrixAlgEq …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem detAux_def'' {ι' : Type*} [Fintype ι'] [DecidableEq ι'] (tb : Trunc <| Basis ι A M)
    (b' : Basis ι' A M) (f : M →ₗ[A] M) :
    LinearMap.detAux tb f = Matrix.det (LinearMap.toMatrix b' b' f) := by
  induction tb using Trunc.induction_on with
  | h b => rw [detAux_def', det_toMatrix_eq_det_toMatrix b b']


@[simp]
theorem detAux_id (b : Trunc <| Basis ι A M) : LinearMap.detAux b LinearMap.id = 1 :=
  (LinearMap.detAux b).map_one


@[simp]
theorem detAux_comp (b : Trunc <| Basis ι A M) (f g : M →ₗ[A] M) :
    LinearMap.detAux b (f.comp g) = LinearMap.detAux b f * LinearMap.detAux b g :=
  (LinearMap.detAux b).map_mul f g


open scoped Classical in
-- Discourage the elaborator from unfolding `det` and producing a huge term by marking it
-- as irreducible.
/-- The determinant of an endomorphism independent of basis.

If there is no finite basis on `M`, the result is `1` instead.
-/
protected irreducible_def det : (M →ₗ[A] M) →* A :=
  if H : ∃ s : Finset M, Nonempty (Basis s A M) then LinearMap.detAux (Trunc.mk H.choose_spec.some)
  else 1


open scoped Classical in
theorem coe_det [DecidableEq M] :
    ⇑(LinearMap.det : (M →ₗ[A] M) →* A) =
      if H : ∃ s : Finset M, Nonempty (Basis s A M) then
        LinearMap.detAux (Trunc.mk H.choose_spec.some)
      else 1 := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    ⊢ Eq ⇑LinearMap.det ⇑(dite (Exists fun s => Nonempty (Basis (Subtype fun x =>  …
  -/
  ext
  /-
    case h
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    x✝ : LinearMap (RingHom.id A) M M
    ⊢ Eq (LinearMap.det x✝) ((dite (Exists fun s => Nonempty (Basis (Subtype fun x …
  -/
  rw [LinearMap.det_def]
  /-
    case h
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    x✝ : LinearMap (RingHom.id A) M M
    ⊢ Eq ((dite (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem  …
  -/
  split_ifs
    /-
      case pos
      M : Type u_2
      inst✝³ : AddCommGroup M
      A : Type u_5
      inst✝² : CommRing A
      inst✝¹ : Module A M
      inst✝ : DecidableEq M
      x✝ : LinearMap (RingHom.id A) M M
      h✝ : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) A M)
      ⊢ Eq ((LinearMap.detAux (Trunc.mk ⋯.some)) x✝) ((LinearMap.detAux (Trunc.mk ⋯. …
    -/
  · congr -- use the correct `DecidableEq` instance
    /-
      🎉 no goals
    -/
  /-
    case neg
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    x✝ : LinearMap (RingHom.id A) M M
    h✝ : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x …
    ⊢ Eq (1 x✝) (1 x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem det_eq_det_toMatrix_of_finset [DecidableEq M] {s : Finset M} (b : Basis s A M)
    (f : M →ₗ[A] M) : LinearMap.det f = Matrix.det (LinearMap.toMatrix b b f) := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) A M
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq (LinearMap.det f) ((LinearMap.toMatrix b b) f).det
  -/
  have : ∃ s : Finset M, Nonempty (Basis s A M) := ⟨s, ⟨b⟩⟩
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) A M
    f : LinearMap (RingHom.id A) M M
    this : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) A …
    ⊢ Eq (LinearMap.det f) ((LinearMap.toMatrix b b) f).det
  -/
  rw [LinearMap.coe_det, dif_pos, detAux_def'' _ b] <;> assumption
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem det_toMatrix (b : Basis ι A M) (f : M →ₗ[A] M) :
    Matrix.det (toMatrix b b f) = LinearMap.det f := by
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b : Basis ι A M
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq ((LinearMap.toMatrix b b) f).det (LinearMap.det f)
  -/
  haveI := Classical.decEq M
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b : Basis ι A M
    f : LinearMap (RingHom.id A) M M
    this : DecidableEq M
    ⊢ Eq ((LinearMap.toMatrix b b) f).det (LinearMap.det f)
  -/
  rw [det_eq_det_toMatrix_of_finset b.reindexFinsetRange]
  -- Porting note: moved out of `rw` due to error
  -- typeclass instance problem is stuck, it is often due to metavariables `DecidableEq ?m.628881`
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b : Basis ι A M
    f : LinearMap (RingHom.id A) M M
    this : DecidableEq M
    ⊢ Eq ((LinearMap.toMatrix b b) f).det ((LinearMap.toMatrix b.reindexFinsetRang …
  -/
  apply det_toMatrix_eq_det_toMatrix b
  /-
    🎉 no goals
  -/


@[simp]
theorem det_toMatrix' {ι : Type*} [Fintype ι] [DecidableEq ι] (f : (ι → A) →ₗ[A] ι → A) :
                                                               /-
                                                                 A : Type u_5
                                                                 inst✝² : CommRing A
                                                                 ι : Type u_7
                                                                 inst✝¹ : Fintype ι
                                                                 inst✝ : DecidableEq ι
                                                                 f : LinearMap (RingHom.id A) (ι → A) (ι → A)
                                                                 ⊢ Eq (LinearMap.toMatrix' f).det (LinearMap.det f)
                                                               -/
    Matrix.det (LinearMap.toMatrix' f) = LinearMap.det f := by simp [← toMatrix_eq_toMatrix']
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem det_toLin (b : Basis ι R M) (f : Matrix ι ι R) :
    LinearMap.det (Matrix.toLin b b f) = f.det := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    f : Matrix ι ι R
    ⊢ Eq (LinearMap.det ((Matrix.toLin b b) f)) f.det
  -/
  rw [← LinearMap.det_toMatrix b, LinearMap.toMatrix_toLin]
  /-
    🎉 no goals
  -/


@[simp]
theorem det_toLin' (f : Matrix ι ι R) : LinearMap.det (Matrix.toLin' f) = Matrix.det f := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : Matrix ι ι R
    ⊢ Eq (LinearMap.det (Matrix.toLin' f)) f.det
  -/
  simp only [← toLin_eq_toLin', det_toLin]
  /-
    🎉 no goals
  -/


/-- To show `P (LinearMap.det f)` it suffices to consider `P (Matrix.det (toMatrix _ _ f))` and
`P 1`. -/
@[elab_as_elim]
theorem det_cases [DecidableEq M] {P : A → Prop} (f : M →ₗ[A] M)
    (hb : ∀ (s : Finset M) (b : Basis s A M), P (Matrix.det (toMatrix b b f))) (h1 : P 1) :
    P (LinearMap.det f) := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    P : A → Prop
    f : LinearMap (RingHom.id A) M M
    hb : ∀ (s : Finset M) (b : Basis (Subtype fun x => Membership.mem s x) A M), P …
    h1 : P 1
    ⊢ P (LinearMap.det f)
  -/
  rw [LinearMap.det_def]
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : DecidableEq M
    P : A → Prop
    f : LinearMap (RingHom.id A) M M
    hb : ∀ (s : Finset M) (b : Basis (Subtype fun x => Membership.mem s x) A M), P …
    h1 : P 1
    ⊢ P ((dite (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s …
  -/
  split_ifs with h
    /-
      case pos
      M : Type u_2
      inst✝³ : AddCommGroup M
      A : Type u_5
      inst✝² : CommRing A
      inst✝¹ : Module A M
      inst✝ : DecidableEq M
      P : A → Prop
      f : LinearMap (RingHom.id A) M M
      hb : ∀ (s : Finset M) (b : Basis (Subtype fun x => Membership.mem s x) A M), P …
      h1 : P 1
      h : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) A M)
      ⊢ P ((LinearMap.detAux (Trunc.mk ⋯.some)) f)
    -/
  · convert hb _ h.choose_spec.some
    -- Porting note: was `apply det_aux_def'`
    /-
      case h.e'_1
      M : Type u_2
      inst✝³ : AddCommGroup M
      A : Type u_5
      inst✝² : CommRing A
      inst✝¹ : Module A M
      inst✝ : DecidableEq M
      P : A → Prop
      f : LinearMap (RingHom.id A) M M
      hb : ∀ (s : Finset M) (b : Basis (Subtype fun x => Membership.mem s x) A M), P …
      h1 : P 1
      h : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) A M)
      ⊢ Eq ((LinearMap.detAux (Trunc.mk ⋯.some)) f) ((LinearMap.toMatrix ⋯.some ⋯.so …
    -/
    convert detAux_def'' (Trunc.mk h.choose_spec.some) h.choose_spec.some f
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_2
      inst✝³ : AddCommGroup M
      A : Type u_5
      inst✝² : CommRing A
      inst✝¹ : Module A M
      inst✝ : DecidableEq M
      P : A → Prop
      f : LinearMap (RingHom.id A) M M
      hb : ∀ (s : Finset M) (b : Basis (Subtype fun x => Membership.mem s x) A M), P …
      h1 : P 1
      h : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ P (1 f)
    -/
  · exact h1
    /-
      🎉 no goals
    -/


@[simp]
theorem det_comp (f g : M →ₗ[A] M) :
    LinearMap.det (f.comp g) = LinearMap.det f * LinearMap.det g :=
  LinearMap.det.map_mul f g


@[simp]
theorem det_id : LinearMap.det (LinearMap.id : M →ₗ[A] M) = 1 :=
  LinearMap.det.map_one


/-- Multiplying a map by a scalar `c` multiplies its determinant by `c ^ dim M`. -/
@[simp]
theorem det_smul [Module.Free A M] (c : A) (f : M →ₗ[A] M) :
    LinearMap.det (c • f) = c ^ Module.finrank A M * LinearMap.det f := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : Module.Free A M
    c : A
    f : LinearMap (RingHom.id A) M M
    ⊢ Eq (LinearMap.det (HSMul.hSMul c f)) (HMul.hMul (HPow.hPow c (Module.finrank …
  -/
  nontriviality A
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : Module.Free A M
    c : A
    f : LinearMap (RingHom.id A) M M
    a✝ : Nontrivial A
    ⊢ Eq (LinearMap.det (HSMul.hSMul c f)) (HMul.hMul (HPow.hPow c (Module.finrank …
  -/
  by_cases H : ∃ s : Finset M, Nonempty (Basis s A M)
  · have : Module.Finite A M := by
      rcases H with ⟨s, ⟨hs⟩⟩
      exact Module.Finite.of_basis hs
    simp only [← det_toMatrix (Module.finBasis A M), LinearEquiv.map_smul,
      Fintype.card_fin, Matrix.det_smul]
  · classical
      have : Module.finrank A M = 0 := finrank_eq_zero_of_not_exists_basis H
      simp [coe_det, H, this]


theorem det_zero' {ι : Type*} [Finite ι] [Nonempty ι] (b : Basis ι A M) :
    LinearMap.det (0 : M →ₗ[A] M) = 0 := by
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    A : Type u_5
    inst✝³ : CommRing A
    inst✝² : Module A M
    ι : Type u_7
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    b : Basis ι A M
    ⊢ Eq (LinearMap.det 0) 0
  -/
  haveI := Classical.decEq ι
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    A : Type u_5
    inst✝³ : CommRing A
    inst✝² : Module A M
    ι : Type u_7
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    b : Basis ι A M
    this : DecidableEq ι
    ⊢ Eq (LinearMap.det 0) 0
  -/
  cases nonempty_fintype ι
  /-
    case intro
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    A : Type u_5
    inst✝³ : CommRing A
    inst✝² : Module A M
    ι : Type u_7
    inst✝¹ : Finite ι
    inst✝ : Nonempty ι
    b : Basis ι A M
    this : DecidableEq ι
    val✝ : Fintype ι
    ⊢ Eq (LinearMap.det 0) 0
  -/
  rwa [← det_toMatrix b, LinearEquiv.map_zero, det_zero]
  /-
    🎉 no goals
  -/


/-- In a finite-dimensional vector space, the zero map has determinant `1` in dimension `0`,
and `0` otherwise. We give a formula that also works in infinite dimension, where we define
the determinant to be `1`. -/
@[simp]
theorem det_zero [Module.Free A M] :
    LinearMap.det (0 : M →ₗ[A] M) = (0 : A) ^ Module.finrank A M := by
  /-
    M : Type u_2
    inst✝³ : AddCommGroup M
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : Module.Free A M
    ⊢ Eq (LinearMap.det 0) (HPow.hPow 0 (Module.finrank A M))
  -/
  simp only [← zero_smul A (1 : M →ₗ[A] M), det_smul, mul_one, MonoidHom.map_one]
  /-
    🎉 no goals
  -/


theorem det_eq_one_of_not_module_finite (h : ¬Module.Finite R M) (f : M →ₗ[R] M) : f.det = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : Not (Module.Finite R M)
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq (LinearMap.det f) 1
  -/
  rw [LinearMap.det, dif_neg, MonoidHom.one_apply]
  /-
    case hnc
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : Not (Module.Finite R M)
    f : LinearMap (RingHom.id R) M M
    ⊢ Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R …
  -/
  exact fun ⟨_, ⟨b⟩⟩ ↦ h (Module.Finite.of_basis b)
  /-
    🎉 no goals
  -/


theorem det_eq_one_of_subsingleton [Subsingleton M] (f : M →ₗ[R] M) :
    LinearMap.det (f : M →ₗ[R] M) = 1 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq (LinearMap.det f) 1
  -/
  have b : Basis (Fin 0) R M := Basis.empty M
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    f : LinearMap (RingHom.id R) M M
    b : Basis (Fin 0) R M
    ⊢ Eq (LinearMap.det f) 1
  -/
  rw [← f.det_toMatrix b]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    f : LinearMap (RingHom.id R) M M
    b : Basis (Fin 0) R M
    ⊢ Eq ((LinearMap.toMatrix b b) f).det 1
  -/
  exact Matrix.det_isEmpty
  /-
    🎉 no goals
  -/


theorem det_eq_one_of_finrank_eq_zero {𝕜 : Type*} [Field 𝕜] {M : Type*} [AddCommGroup M]
    [Module 𝕜 M] (h : Module.finrank 𝕜 M = 0) (f : M →ₗ[𝕜] M) :
    LinearMap.det (f : M →ₗ[𝕜] M) = 1 := by
  classical
    refine @LinearMap.det_cases M _ 𝕜 _ _ _ (fun t => t = 1) f ?_ rfl
    intro s b
    have : IsEmpty s := by
      rw [← Fintype.card_eq_zero_iff]
      exact (Module.finrank_eq_card_basis b).symm.trans h
    exact Matrix.det_isEmpty


/-- Conjugating a linear map by a linear equiv does not change its determinant. -/
@[simp]
theorem det_conj {N : Type*} [AddCommGroup N] [Module A N] (f : M →ₗ[A] M) (e : M ≃ₗ[A] N) :
    LinearMap.det ((e : M →ₗ[A] N) ∘ₗ f ∘ₗ (e.symm : N →ₗ[A] M)) = LinearMap.det f := by
  classical
    by_cases H : ∃ s : Finset M, Nonempty (Basis s A M)
    · rcases H with ⟨s, ⟨b⟩⟩
      rw [← det_toMatrix b f, ← det_toMatrix (b.map e), toMatrix_comp (b.map e) b (b.map e),
        toMatrix_comp (b.map e) b b, ← Matrix.mul_assoc, Matrix.det_conj_of_mul_eq_one]
      · rw [← toMatrix_comp, LinearEquiv.comp_coe, e.symm_trans_self, LinearEquiv.refl_toLinearMap,
          toMatrix_id]
      · rw [← toMatrix_comp, LinearEquiv.comp_coe, e.self_trans_symm, LinearEquiv.refl_toLinearMap,
          toMatrix_id]
    · have H' : ¬∃ t : Finset N, Nonempty (Basis t A N) := by
        contrapose! H
        rcases H with ⟨s, ⟨b⟩⟩
        exact ⟨_, ⟨(b.map e.symm).reindexFinsetRange⟩⟩
      simp only [coe_det, H, H', MonoidHom.one_apply, dif_neg, not_false_eq_true]


/-- If a linear map is invertible, so is its determinant. -/
theorem isUnit_det {A : Type*} [CommRing A] [Module A M] (f : M →ₗ[A] M) (hf : IsUnit f) :
    IsUnit (LinearMap.det f) := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    A : Type u_7
    inst✝¹ : CommRing A
    inst✝ : Module A M
    f : LinearMap (RingHom.id A) M M
    hf : IsUnit f
    ⊢ IsUnit (LinearMap.det f)
  -/
  obtain ⟨g, hg⟩ : ∃ g, f.comp g = 1 := hf.exists_right_inv
  have : LinearMap.det f * LinearMap.det g = 1 := by
    simp only [← LinearMap.det_comp, hg, MonoidHom.map_one]
  /-
    case intro
    M : Type u_2
    inst✝² : AddCommGroup M
    A : Type u_7
    inst✝¹ : CommRing A
    inst✝ : Module A M
    f : LinearMap (RingHom.id A) M M
    hf : IsUnit f
    g : LinearMap (RingHom.id A) M M
    hg : Eq (f.comp g) 1
    this : Eq (HMul.hMul (LinearMap.det f) (LinearMap.det g)) 1
    ⊢ IsUnit (LinearMap.det f)
  -/
  exact isUnit_of_mul_eq_one _ _ this
  /-
    🎉 no goals
  -/


/-- If a linear map has determinant different from `1`, then the space is finite-dimensional. -/
theorem finiteDimensional_of_det_ne_one {𝕜 : Type*} [Field 𝕜] [Module 𝕜 M] (f : M →ₗ[𝕜] M)
    (hf : LinearMap.det f ≠ 1) : FiniteDimensional 𝕜 M := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    hf : Ne (LinearMap.det f) 1
    ⊢ FiniteDimensional 𝕜 M
  -/
  by_cases H : ∃ s : Finset M, Nonempty (Basis s 𝕜 M)
    /-
      case pos
      M : Type u_2
      inst✝² : AddCommGroup M
      𝕜 : Type u_7
      inst✝¹ : Field 𝕜
      inst✝ : Module 𝕜 M
      f : LinearMap (RingHom.id 𝕜) M M
      hf : Ne (LinearMap.det f) 1
      H : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) 𝕜 M)
      ⊢ FiniteDimensional 𝕜 M
    -/
  · rcases H with ⟨s, ⟨hs⟩⟩
    /-
      case pos.intro.intro
      M : Type u_2
      inst✝² : AddCommGroup M
      𝕜 : Type u_7
      inst✝¹ : Field 𝕜
      inst✝ : Module 𝕜 M
      f : LinearMap (RingHom.id 𝕜) M M
      hf : Ne (LinearMap.det f) 1
      s : Finset M
      hs : Basis (Subtype fun x => Membership.mem s x) 𝕜 M
      ⊢ FiniteDimensional 𝕜 M
    -/
    exact FiniteDimensional.of_fintype_basis hs
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_2
      inst✝² : AddCommGroup M
      𝕜 : Type u_7
      inst✝¹ : Field 𝕜
      inst✝ : Module 𝕜 M
      f : LinearMap (RingHom.id 𝕜) M M
      hf : Ne (LinearMap.det f) 1
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ FiniteDimensional 𝕜 M
    -/
  · classical simp [LinearMap.coe_det, H] at hf
    /-
      🎉 no goals
    -/


/-- If the determinant of a map vanishes, then the map is not onto. -/
theorem range_lt_top_of_det_eq_zero {𝕜 : Type*} [Field 𝕜] [Module 𝕜 M] {f : M →ₗ[𝕜] M}
    (hf : LinearMap.det f = 0) : LinearMap.range f < ⊤ := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    hf : Eq (LinearMap.det f) 0
    ⊢ LT.lt (LinearMap.range f) Top.top
  -/
  have : FiniteDimensional 𝕜 M := by simp [f.finiteDimensional_of_det_ne_one, hf]
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    hf : Eq (LinearMap.det f) 0
    this : FiniteDimensional 𝕜 M
    ⊢ LT.lt (LinearMap.range f) Top.top
  -/
  contrapose hf
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    this : FiniteDimensional 𝕜 M
    hf : Not (LT.lt (LinearMap.range f) Top.top)
    ⊢ Not (Eq (LinearMap.det f) 0)
  -/
  simp only [lt_top_iff_ne_top, Classical.not_not, ← isUnit_iff_range_eq_top] at hf
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    this : FiniteDimensional 𝕜 M
    hf : IsUnit f
    ⊢ Not (Eq (LinearMap.det f) 0)
  -/
  exact isUnit_iff_ne_zero.1 (f.isUnit_det hf)
  /-
    🎉 no goals
  -/


/-- If the determinant of a map vanishes, then the map is not injective. -/
theorem bot_lt_ker_of_det_eq_zero {𝕜 : Type*} [Field 𝕜] [Module 𝕜 M] {f : M →ₗ[𝕜] M}
    (hf : LinearMap.det f = 0) : ⊥ < LinearMap.ker f := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    hf : Eq (LinearMap.det f) 0
    ⊢ LT.lt Bot.bot (LinearMap.ker f)
  -/
  have : FiniteDimensional 𝕜 M := by simp [f.finiteDimensional_of_det_ne_one, hf]
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    hf : Eq (LinearMap.det f) 0
    this : FiniteDimensional 𝕜 M
    ⊢ LT.lt Bot.bot (LinearMap.ker f)
  -/
  contrapose hf
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    this : FiniteDimensional 𝕜 M
    hf : Not (LT.lt Bot.bot (LinearMap.ker f))
    ⊢ Not (Eq (LinearMap.det f) 0)
  -/
  simp only [bot_lt_iff_ne_bot, Classical.not_not, ← isUnit_iff_ker_eq_bot] at hf
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_7
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearMap (RingHom.id 𝕜) M M
    this : FiniteDimensional 𝕜 M
    hf : IsUnit f
    ⊢ Not (Eq (LinearMap.det f) 0)
  -/
  exact isUnit_iff_ne_zero.1 (f.isUnit_det hf)
  /-
    🎉 no goals
  -/


/-- When the function is over the base ring, the determinant is the evaluation at `1`. -/
@[simp] lemma det_ring (f : R →ₗ[R] R) : f.det = f 1 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : LinearMap (RingHom.id R) R R
    ⊢ Eq (LinearMap.det f) (f 1)
  -/
  simp [← det_toMatrix (Basis.singleton Unit R)]
  /-
    🎉 no goals
  -/


                                                        /-
                                                          R : Type u_1
                                                          inst✝ : CommRing R
                                                          a : R
                                                          ⊢ Eq (LinearMap.det (LinearMap.mulLeft R a)) a
                                                        -/
lemma det_mulLeft (a : R) : (mulLeft R a).det = a := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/

                                                          /-
                                                            R : Type u_1
                                                            inst✝ : CommRing R
                                                            a : R
                                                            ⊢ Eq (LinearMap.det (LinearMap.mulRight R a)) a
                                                          -/
lemma det_mulRight (a : R) : (mulRight R a).det = a := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- On a `LinearEquiv`, the domain of `LinearMap.det` can be promoted to `Rˣ`. -/
protected def det : (M ≃ₗ[R] M) →* Rˣ :=
  (Units.map (LinearMap.det : (M →ₗ[R] M) →* R)).comp
    (LinearMap.GeneralLinearGroup.generalLinearEquiv R M).symm.toMonoidHom


@[simp]
theorem coe_det (f : M ≃ₗ[R] M) : ↑(LinearEquiv.det f) = LinearMap.det (f : M →ₗ[R] M) :=
  rfl


@[simp]
theorem coe_inv_det (f : M ≃ₗ[R] M) : ↑(LinearEquiv.det f)⁻¹ = LinearMap.det (f.symm : M →ₗ[R] M) :=
  rfl


@[simp]
theorem det_refl : LinearEquiv.det (LinearEquiv.refl R M) = 1 :=
  Units.ext <| LinearMap.det_id


@[simp]
theorem det_trans (f g : M ≃ₗ[R] M) :
    LinearEquiv.det (f.trans g) = LinearEquiv.det g * LinearEquiv.det f :=
  map_mul _ g f


@[simp]
theorem det_symm (f : M ≃ₗ[R] M) : LinearEquiv.det f.symm = LinearEquiv.det f⁻¹ :=
  map_inv _ f


/-- Conjugating a linear equiv by a linear equiv does not change its determinant. -/
@[simp]
theorem det_conj (f : M ≃ₗ[R] M) (e : M ≃ₗ[R] M') :
    LinearEquiv.det ((e.symm.trans f).trans e) = LinearEquiv.det f := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearEquiv (RingHom.id R) M M
    e : LinearEquiv (RingHom.id R) M M'
    ⊢ Eq (LinearEquiv.det ((e.symm.trans f).trans e)) (LinearEquiv.det f)
  -/
  rw [← Units.eq_iff, coe_det, coe_det, ← comp_coe, ← comp_coe, LinearMap.det_conj]
  /-
    🎉 no goals
  -/


/-- The determinants of a `LinearEquiv` and its inverse multiply to 1. -/
@[simp]
theorem LinearEquiv.det_mul_det_symm {A : Type*} [CommRing A] [Module A M] (f : M ≃ₗ[A] M) :
    LinearMap.det (f : M →ₗ[A] M) * LinearMap.det (f.symm : M →ₗ[A] M) = 1 := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    f : LinearEquiv (RingHom.id A) M M
    ⊢ Eq (HMul.hMul (LinearMap.det ↑f) (LinearMap.det ↑f.symm)) 1
  -/
  simp [← LinearMap.det_comp]
  /-
    🎉 no goals
  -/


/-- The determinants of a `LinearEquiv` and its inverse multiply to 1. -/
@[simp]
theorem LinearEquiv.det_symm_mul_det {A : Type*} [CommRing A] [Module A M] (f : M ≃ₗ[A] M) :
    LinearMap.det (f.symm : M →ₗ[A] M) * LinearMap.det (f : M →ₗ[A] M) = 1 := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    f : LinearEquiv (RingHom.id A) M M
    ⊢ Eq (HMul.hMul (LinearMap.det ↑f.symm) (LinearMap.det ↑f)) 1
  -/
  simp [← LinearMap.det_comp]
  /-
    🎉 no goals
  -/

-- Cannot be stated using `LinearMap.det` because `f` is not an endomorphism.

theorem LinearEquiv.isUnit_det (f : M ≃ₗ[R] M') (v : Basis ι R M) (v' : Basis ι R M') :
    IsUnit (LinearMap.toMatrix v v' f).det := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearEquiv (RingHom.id R) M M'
    v : Basis ι R M
    v' : Basis ι R M'
    ⊢ IsUnit ((LinearMap.toMatrix v v') ↑f).det
  -/
  apply isUnit_det_of_left_inverse
  /-
    case h
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearEquiv (RingHom.id R) M M'
    v : Basis ι R M
    v' : Basis ι R M'
    ⊢ Eq (HMul.hMul ?B ((LinearMap.toMatrix v v') ↑f)) 1
  -/
  simpa using (LinearMap.toMatrix_comp v v' v f.symm f).symm
  /-
    🎉 no goals
  -/


/-- Specialization of `LinearEquiv.isUnit_det` -/
theorem LinearEquiv.isUnit_det' {A : Type*} [CommRing A] [Module A M] (f : M ≃ₗ[A] M) :
    IsUnit (LinearMap.det (f : M →ₗ[A] M)) :=
  isUnit_of_mul_eq_one _ _ f.det_mul_det_symm


/-- The determinant of `f.symm` is the inverse of that of `f` when `f` is a linear equiv. -/
theorem LinearEquiv.det_coe_symm {𝕜 : Type*} [Field 𝕜] [Module 𝕜 M] (f : M ≃ₗ[𝕜] M) :
    LinearMap.det (f.symm : M →ₗ[𝕜] M) = (LinearMap.det (f : M →ₗ[𝕜] M))⁻¹ := by
  /-
    M : Type u_2
    inst✝² : AddCommGroup M
    𝕜 : Type u_5
    inst✝¹ : Field 𝕜
    inst✝ : Module 𝕜 M
    f : LinearEquiv (RingHom.id 𝕜) M M
    ⊢ Eq (LinearMap.det ↑f.symm) (Inv.inv (LinearMap.det ↑f))
  -/
  field_simp [IsUnit.ne_zero f.isUnit_det']
  /-
    🎉 no goals
  -/


/-- Builds a linear equivalence from a linear map whose determinant in some bases is a unit. -/
@[simps]
def LinearEquiv.ofIsUnitDet {f : M →ₗ[R] M'} {v : Basis ι R M} {v' : Basis ι R M'}
    (h : IsUnit (LinearMap.toMatrix v v' f).det) : M ≃ₗ[R] M' where
  toFun := f
  map_add' := f.map_add
  map_smul' := f.map_smul
  invFun := toLin v' v (toMatrix v v' f)⁻¹
  left_inv x :=
    calc toLin v' v (toMatrix v v' f)⁻¹ (f x)
      _ = toLin v v ((toMatrix v v' f)⁻¹ * toMatrix v v' f) x := by
        /-
          R : Type u_1
          inst✝⁶ : CommRing R
          M : Type u_2
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          M' : Type u_3
          inst✝³ : AddCommGroup M'
          inst✝² : Module R M'
          ι : Type u_4
          inst✝¹ : DecidableEq ι
          inst✝ : Fintype ι
          e : Basis ι R M
          f : LinearMap (RingHom.id R) M M'
          v : Basis ι R M
          v' : Basis ι R M'
          h : IsUnit ((LinearMap.toMatrix v v') f).det
          x : M
          ⊢ Eq (((Matrix.toLin v' v) (Inv.inv ((LinearMap.toMatrix v v') f))) (f x)) ((( …
        -/
        rw [toLin_mul v v' v, toLin_toMatrix, LinearMap.comp_apply]
        /-
          🎉 no goals
        -/
                  /-
                    R : Type u_1
                    inst✝⁶ : CommRing R
                    M : Type u_2
                    inst✝⁵ : AddCommGroup M
                    inst✝⁴ : Module R M
                    M' : Type u_3
                    inst✝³ : AddCommGroup M'
                    inst✝² : Module R M'
                    ι : Type u_4
                    inst✝¹ : DecidableEq ι
                    inst✝ : Fintype ι
                    e : Basis ι R M
                    f : LinearMap (RingHom.id R) M M'
                    v : Basis ι R M
                    v' : Basis ι R M'
                    h : IsUnit ((LinearMap.toMatrix v v') f).det
                    x : M
                    ⊢ Eq (((Matrix.toLin v v) (HMul.hMul (Inv.inv ((LinearMap.toMatrix v v') f)) ( …
                  -/
      _ = x := by simp [h]
                  /-
                    🎉 no goals
                  -/
  right_inv x :=
    calc f (toLin v' v (toMatrix v v' f)⁻¹ x)
      _ = toLin v' v' (toMatrix v v' f * (toMatrix v v' f)⁻¹) x := by
        /-
          R : Type u_1
          inst✝⁶ : CommRing R
          M : Type u_2
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          M' : Type u_3
          inst✝³ : AddCommGroup M'
          inst✝² : Module R M'
          ι : Type u_4
          inst✝¹ : DecidableEq ι
          inst✝ : Fintype ι
          e : Basis ι R M
          f : LinearMap (RingHom.id R) M M'
          v : Basis ι R M
          v' : Basis ι R M'
          h : IsUnit ((LinearMap.toMatrix v v') f).det
          x : M'
          ⊢ Eq (f (((Matrix.toLin v' v) (Inv.inv ((LinearMap.toMatrix v v') f))) x)) ((( …
        -/
        rw [toLin_mul v' v v', LinearMap.comp_apply, toLin_toMatrix v v']
        /-
          🎉 no goals
        -/
                  /-
                    R : Type u_1
                    inst✝⁶ : CommRing R
                    M : Type u_2
                    inst✝⁵ : AddCommGroup M
                    inst✝⁴ : Module R M
                    M' : Type u_3
                    inst✝³ : AddCommGroup M'
                    inst✝² : Module R M'
                    ι : Type u_4
                    inst✝¹ : DecidableEq ι
                    inst✝ : Fintype ι
                    e : Basis ι R M
                    f : LinearMap (RingHom.id R) M M'
                    v : Basis ι R M
                    v' : Basis ι R M'
                    h : IsUnit ((LinearMap.toMatrix v v') f).det
                    x : M'
                    ⊢ Eq (((Matrix.toLin v' v') (HMul.hMul ((LinearMap.toMatrix v v') f) (Inv.inv  …
                  -/
      _ = x := by simp [h]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem LinearEquiv.coe_ofIsUnitDet {f : M →ₗ[R] M'} {v : Basis ι R M} {v' : Basis ι R M'}
    (h : IsUnit (LinearMap.toMatrix v v' f).det) :
    (LinearEquiv.ofIsUnitDet h : M →ₗ[R] M') = f := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) M M'
    v : Basis ι R M
    v' : Basis ι R M'
    h : IsUnit ((LinearMap.toMatrix v v') f).det
    ⊢ Eq (↑(LinearEquiv.ofIsUnitDet h)) f
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) M M'
    v : Basis ι R M
    v' : Basis ι R M'
    h : IsUnit ((LinearMap.toMatrix v v') f).det
    x : M
    ⊢ Eq (↑(LinearEquiv.ofIsUnitDet h) x) (f x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Builds a linear equivalence from a linear map on a finite-dimensional vector space whose
determinant is nonzero. -/
abbrev LinearMap.equivOfDetNeZero {𝕜 : Type*} [Field 𝕜] {M : Type*} [AddCommGroup M] [Module 𝕜 M]
    [FiniteDimensional 𝕜 M] (f : M →ₗ[𝕜] M) (hf : LinearMap.det f ≠ 0) : M ≃ₗ[𝕜] M :=
  have : IsUnit (LinearMap.toMatrix (Module.finBasis 𝕜 M)
      (Module.finBasis 𝕜 M) f).det := by
    /-
      R : Type u_1
      inst✝¹⁰ : CommRing R
      M✝ : Type u_2
      inst✝⁹ : AddCommGroup M✝
      inst✝⁸ : Module R M✝
      M' : Type u_3
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      ι : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Fintype ι
      e : Basis ι R M✝
      𝕜 : Type u_5
      inst✝³ : Field 𝕜
      M : Type u_6
      inst✝² : AddCommGroup M
      inst✝¹ : Module 𝕜 M
      inst✝ : FiniteDimensional 𝕜 M
      f : LinearMap (RingHom.id 𝕜) M M
      hf : Ne (LinearMap.det f) 0
      ⊢ IsUnit ((LinearMap.toMatrix (Module.finBasis 𝕜 M) (Module.finBasis 𝕜 M)) f). …
    -/
    rw [LinearMap.det_toMatrix]
    /-
      R : Type u_1
      inst✝¹⁰ : CommRing R
      M✝ : Type u_2
      inst✝⁹ : AddCommGroup M✝
      inst✝⁸ : Module R M✝
      M' : Type u_3
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      ι : Type u_4
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Fintype ι
      e : Basis ι R M✝
      𝕜 : Type u_5
      inst✝³ : Field 𝕜
      M : Type u_6
      inst✝² : AddCommGroup M
      inst✝¹ : Module 𝕜 M
      inst✝ : FiniteDimensional 𝕜 M
      f : LinearMap (RingHom.id 𝕜) M M
      hf : Ne (LinearMap.det f) 0
      ⊢ IsUnit (LinearMap.det f)
    -/
    exact isUnit_iff_ne_zero.2 hf
    /-
      🎉 no goals
    -/
  LinearEquiv.ofIsUnitDet this


theorem LinearMap.associated_det_of_eq_comp (e : M ≃ₗ[R] M) (f f' : M →ₗ[R] M)
    (h : ∀ x, f x = f' (e x)) : Associated (LinearMap.det f) (LinearMap.det f') := by
  suffices Associated (LinearMap.det (f' ∘ₗ ↑e)) (LinearMap.det f') by
    convert this using 2
    ext x
    exact h x
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : LinearEquiv (RingHom.id R) M M
    f f' : LinearMap (RingHom.id R) M M
    h : ∀ (x : M), Eq (f x) (f' (e x))
    ⊢ Associated (LinearMap.det (f'.comp ↑e)) (LinearMap.det f')
  -/
  rw [← mul_one (LinearMap.det f'), LinearMap.det_comp]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : LinearEquiv (RingHom.id R) M M
    f f' : LinearMap (RingHom.id R) M M
    h : ∀ (x : M), Eq (f x) (f' (e x))
    ⊢ Associated (HMul.hMul (LinearMap.det f') (LinearMap.det ↑e)) (HMul.hMul (Lin …
  -/
  exact Associated.mul_left _ (associated_one_iff_isUnit.mpr e.isUnit_det')
  /-
    🎉 no goals
  -/


theorem LinearMap.associated_det_comp_equiv {N : Type*} [AddCommGroup N] [Module R N]
    (f : N →ₗ[R] M) (e e' : M ≃ₗ[R] N) :
    Associated (LinearMap.det (f ∘ₗ ↑e)) (LinearMap.det (f ∘ₗ ↑e')) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) N M
    e e' : LinearEquiv (RingHom.id R) M N
    ⊢ Associated (LinearMap.det (f.comp ↑e)) (LinearMap.det (f.comp ↑e'))
  -/
  refine LinearMap.associated_det_of_eq_comp (e.trans e'.symm) _ _ ?_
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_5
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) N M
    e e' : LinearEquiv (RingHom.id R) M N
    ⊢ ∀ (x : M), Eq ((f.comp ↑e) x) ((f.comp ↑e') ((e.trans e'.symm) x))
  -/
  intro x
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, LinearEquiv.trans_apply,
    LinearEquiv.apply_symm_apply]


/-- The determinant of a family of vectors with respect to some basis, as an alternating
multilinear map. -/
nonrec def Basis.det : M [⋀^ι]→ₗ[R] R where
  toFun v := det (e.toMatrix v)
  map_update_add' := by
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (x y : M), Eq ((fun v => (e.toM …
    -/
    intro inst v i x y
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      v : ι → M
      i : ι
      x y : M
      ⊢ Eq ((fun v => (e.toMatrix v).det) (Function.update v i (HAdd.hAdd x y))) (HA …
    -/
    cases Subsingleton.elim inst ‹_›
    /-
      case refl
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      v : ι → M
      i : ι
      x y : M
      ⊢ Eq ((fun v => (e.toMatrix v).det) (Function.update v i (HAdd.hAdd x y))) (HA …
    -/
    simp only [e.toMatrix_update, LinearEquiv.map_add, Finsupp.coe_add]
    -- Porting note: was `exact det_update_column_add _ _ _ _`
    /-
      case refl
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      v : ι → M
      i : ι
      x y : M
      ⊢ Eq ((e.toMatrix v).updateCol i (HAdd.hAdd ⇑(e.repr x) ⇑(e.repr y))).det (HAd …
    -/
    convert det_updateCol_add (e.toMatrix v) i (e.repr x) (e.repr y)
    /-
      🎉 no goals
    -/
  map_update_smul' := by
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x : M), Eq ((fun v =>  …
    -/
    intro inst u i c x
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      u : ι → M
      i : ι
      c : R
      x : M
      ⊢ Eq ((fun v => (e.toMatrix v).det) (Function.update u i (HSMul.hSMul c x))) ( …
    -/
    cases Subsingleton.elim inst ‹_›
    /-
      case refl
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      u : ι → M
      i : ι
      c : R
      x : M
      ⊢ Eq ((fun v => (e.toMatrix v).det) (Function.update u i (HSMul.hSMul c x))) ( …
    -/
    simp only [e.toMatrix_update, Algebra.id.smul_eq_mul, LinearEquiv.map_smul]
    -- Porting note: was `apply det_update_column_smul`
    /-
      case refl
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      inst : DecidableEq ι
      u : ι → M
      i : ι
      c : R
      x : M
      ⊢ Eq ((e.toMatrix u).updateCol i ⇑(HSMul.hSMul c (e.repr x))).det (HMul.hMul c …
    -/
    convert det_updateCol_smul (e.toMatrix u) i c (e.repr x)
    /-
      🎉 no goals
    -/
  map_eq_zero_of_eq' := by
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      ⊢ ∀ (v : ι → M) (i j : ι), Eq (v i) (v j) → Ne i j → Eq ({ toFun := fun v => ( …
    -/
    intro v i j h hij
    -- Porting note: added
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      i j : ι
      h : Eq (v i) (v j)
      hij : Ne i j
      ⊢ Eq ({ toFun := fun v => (e.toMatrix v).det, map_update_add' := ⋯, map_update …
    -/
    simp only
    rw [← Function.update_eq_self i v, h, ← det_transpose, e.toMatrix_update, ← updateRow_transpose,
      ← e.toMatrix_transpose_apply]
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      i j : ι
      h : Eq (v i) (v j)
      hij : Ne i j
      ⊢ Eq ((e.toMatrix v).transpose.updateRow i ((e.toMatrix v).transpose j)).det 0
    -/
    apply det_zero_of_row_eq hij
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      M : Type u_2
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      M' : Type u_3
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      i j : ι
      h : Eq (v i) (v j)
      hij : Ne i j
      ⊢ Eq ((e.toMatrix v).transpose.updateRow i ((e.toMatrix v).transpose j) i) ((e …
    -/
    rw [updateRow_ne hij.symm, updateRow_self]
    /-
      🎉 no goals
    -/


theorem Basis.det_apply (v : ι → M) : e.det v = Matrix.det (e.toMatrix v) :=
  rfl


                                           /-
                                             R : Type u_1
                                             inst✝⁴ : CommRing R
                                             M : Type u_2
                                             inst✝³ : AddCommGroup M
                                             inst✝² : Module R M
                                             ι : Type u_4
                                             inst✝¹ : DecidableEq ι
                                             inst✝ : Fintype ι
                                             e : Basis ι R M
                                             ⊢ Eq (e.det ⇑e) 1
                                           -/
theorem Basis.det_self : e.det e = 1 := by simp [e.det_apply]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem Basis.det_isEmpty [IsEmpty ι] : e.det = AlternatingMap.constOfIsEmpty R M ι 1 := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    e : Basis ι R M
    inst✝ : IsEmpty ι
    ⊢ Eq e.det (AlternatingMap.constOfIsEmpty R M ι 1)
  -/
  ext v
  /-
    case H
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_4
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    e : Basis ι R M
    inst✝ : IsEmpty ι
    v : ι → M
    ⊢ Eq (e.det v) ((AlternatingMap.constOfIsEmpty R M ι 1) v)
  -/
  exact Matrix.det_isEmpty
  /-
    🎉 no goals
  -/


/-- `Basis.det` is not the zero map. -/
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝⁵ : CommRing R
                                                                      M : Type u_2
                                                                      inst✝⁴ : AddCommGroup M
                                                                      inst✝³ : Module R M
                                                                      ι : Type u_4
                                                                      inst✝² : DecidableEq ι
                                                                      inst✝¹ : Fintype ι
                                                                      e : Basis ι R M
                                                                      inst✝ : Nontrivial R
                                                                      h : Eq e.det 0
                                                                      ⊢ False
                                                                    -/
theorem Basis.det_ne_zero [Nontrivial R] : e.det ≠ 0 := fun h => by simpa [h] using e.det_self
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem Basis.smul_det {G} [Group G] [DistribMulAction G M] [SMulCommClass G R M]
    (g : G) (v : ι → M) :
    (g • e).det v = e.det (g⁻¹ • v) := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    M : Type u_2
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    ι : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    e : Basis ι R M
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : DistribMulAction G M
    inst✝ : SMulCommClass G R M
    g : G
    v : ι → M
    ⊢ Eq ((HSMul.hSMul g e).det v) (e.det (HSMul.hSMul (Inv.inv g) v))
  -/
  simp_rw [det_apply, toMatrix_smul_left]
  /-
    🎉 no goals
  -/


theorem is_basis_iff_det {v : ι → M} :
    LinearIndependent R v ∧ span R (Set.range v) = ⊤ ↔ IsUnit (e.det v) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    v : ι → M
    ⊢ Iff (And (LinearIndependent R v) (Eq (Submodule.span R (Set.range v)) Top.to …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      ⊢ And (LinearIndependent R v) (Eq (Submodule.span R (Set.range v)) Top.top) →  …
    -/
  · rintro ⟨hli, hspan⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hspan : Eq (Submodule.span R (Set.range v)) Top.top
      ⊢ IsUnit (e.det v)
    -/
    set v' := Basis.mk hli hspan.ge
    /-
      case mp.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hspan : Eq (Submodule.span R (Set.range v)) Top.top
      v' : Basis ι R M := Basis.mk hli ⋯
      ⊢ IsUnit (e.det v)
    -/
    rw [e.det_apply]
    /-
      case mp.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hspan : Eq (Submodule.span R (Set.range v)) Top.top
      v' : Basis ι R M := Basis.mk hli ⋯
      ⊢ IsUnit (e.toMatrix v).det
    -/
    convert LinearEquiv.isUnit_det (LinearEquiv.refl R M) v' e using 2
    /-
      case h.e'_3.h.e'_6
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hspan : Eq (Submodule.span R (Set.range v)) Top.top
      v' : Basis ι R M := Basis.mk hli ⋯
      ⊢ Eq (e.toMatrix v) ((LinearMap.toMatrix v' e) ↑(LinearEquiv.refl R M))
    -/
    ext i j
    /-
      case h.e'_3.h.e'_6.a
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hspan : Eq (Submodule.span R (Set.range v)) Top.top
      v' : Basis ι R M := Basis.mk hli ⋯
      i j : ι
      ⊢ Eq (e.toMatrix v i j) ((LinearMap.toMatrix v' e) (↑(LinearEquiv.refl R M)) i …
    -/
    simp [v']
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      ⊢ IsUnit (e.det v) → And (LinearIndependent R v) (Eq (Submodule.span R (Set.ra …
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      h : IsUnit (e.det v)
      ⊢ And (LinearIndependent R v) (Eq (Submodule.span R (Set.range v)) Top.top)
    -/
    rw [Basis.det_apply, Basis.toMatrix_eq_toMatrix_constr] at h
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      h : IsUnit ((LinearMap.toMatrix e e) ((e.constr Nat) v)).det
      ⊢ And (LinearIndependent R v) (Eq (Submodule.span R (Set.range v)) Top.top)
    -/
    set v' := Basis.map e (LinearEquiv.ofIsUnitDet h) with v'_def
    have : ⇑v' = v := by
      ext i
      rw [v'_def, Basis.map_apply, LinearEquiv.ofIsUnitDet_apply, e.constr_basis]
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      h : IsUnit ((LinearMap.toMatrix e e) ((e.constr Nat) v)).det
      v' : Basis ι R M := e.map (LinearEquiv.ofIsUnitDet h)
      v'_def : Eq v' (e.map (LinearEquiv.ofIsUnitDet h))
      this : Eq (⇑v') v
      ⊢ And (LinearIndependent R v) (Eq (Submodule.span R (Set.range v)) Top.top)
    -/
    rw [← this]
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      h : IsUnit ((LinearMap.toMatrix e e) ((e.constr Nat) v)).det
      v' : Basis ι R M := e.map (LinearEquiv.ofIsUnitDet h)
      v'_def : Eq v' (e.map (LinearEquiv.ofIsUnitDet h))
      this : Eq (⇑v') v
      ⊢ And (LinearIndependent R ⇑v') (Eq (Submodule.span R (Set.range ⇑v')) Top.top)
    -/
    exact ⟨v'.linearIndependent, v'.span_eq⟩
    /-
      🎉 no goals
    -/


theorem Basis.isUnit_det (e' : Basis ι R M) : IsUnit (e.det e') :=
  (is_basis_iff_det e).mp ⟨e'.linearIndependent, e'.span_eq⟩


/-- Any alternating map to `R` where `ι` has the cardinality of a basis equals the determinant
map with respect to that basis, multiplied by the value of that alternating map on that basis. -/
theorem AlternatingMap.eq_smul_basis_det (f : M [⋀^ι]→ₗ[R] R) : f = f e • e.det := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    f : AlternatingMap R M R ι
    ⊢ Eq f (HSMul.hSMul (f ⇑e) e.det)
  -/
  refine Basis.ext_alternating e fun i h => ?_
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    f : AlternatingMap R M R ι
    i : ι → ι
    h : Function.Injective i
    ⊢ Eq (f fun i_1 => e (i i_1)) ((HSMul.hSMul (f ⇑e) e.det) fun i_1 => e (i i_1))
  -/
  let σ : Equiv.Perm ι := Equiv.ofBijective i (Finite.injective_iff_bijective.1 h)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    f : AlternatingMap R M R ι
    i : ι → ι
    h : Function.Injective i
    σ : Equiv.Perm ι := Equiv.ofBijective i ⋯
    ⊢ Eq (f fun i_1 => e (i i_1)) ((HSMul.hSMul (f ⇑e) e.det) fun i_1 => e (i i_1))
  -/
  change f (e ∘ σ) = (f e • e.det) (e ∘ σ)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    f : AlternatingMap R M R ι
    i : ι → ι
    h : Function.Injective i
    σ : Equiv.Perm ι := Equiv.ofBijective i ⋯
    ⊢ Eq (f (Function.comp ⇑e ⇑σ)) ((HSMul.hSMul (f ⇑e) e.det) (Function.comp ⇑e ⇑ …
  -/
  simp [AlternatingMap.map_perm, Basis.det_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem AlternatingMap.map_basis_eq_zero_iff {ι : Type*} [Finite ι] (e : Basis ι R M)
    (f : M [⋀^ι]→ₗ[R] R) : f e = 0 ↔ f = 0 :=
  ⟨fun h => by
    /-
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type u_5
      inst✝ : Finite ι
      e : Basis ι R M
      f : AlternatingMap R M R ι
      h : Eq (f ⇑e) 0
      ⊢ Eq f 0
    -/
    cases nonempty_fintype ι
    /-
      case intro
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type u_5
      inst✝ : Finite ι
      e : Basis ι R M
      f : AlternatingMap R M R ι
      h : Eq (f ⇑e) 0
      val✝ : Fintype ι
      ⊢ Eq f 0
    -/
    letI := Classical.decEq ι
    /-
      case intro
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type u_5
      inst✝ : Finite ι
      e : Basis ι R M
      f : AlternatingMap R M R ι
      h : Eq (f ⇑e) 0
      val✝ : Fintype ι
      this : DecidableEq ι := Classical.decEq ι
      ⊢ Eq f 0
    -/
    simpa [h] using f.eq_smul_basis_det e,
    /-
      🎉 no goals
    -/
   fun h => h.symm ▸ AlternatingMap.zero_apply _⟩


theorem AlternatingMap.map_basis_ne_zero_iff {ι : Type*} [Finite ι] (e : Basis ι R M)
    (f : M [⋀^ι]→ₗ[R] R) : f e ≠ 0 ↔ f ≠ 0 :=
  not_congr <| f.map_basis_eq_zero_iff e


@[simp]
theorem Basis.det_comp (e : Basis ι A M) (f : M →ₗ[A] M) (v : ι → M) :
    e.det (f ∘ v) = (LinearMap.det f) * e.det v := by
  rw [Basis.det_apply, Basis.det_apply, ← f.det_toMatrix e, ← Matrix.det_mul,
    e.toMatrix_eq_toMatrix_constr (f ∘ v), e.toMatrix_eq_toMatrix_constr v, ← toMatrix_comp,
    e.constr_comp]


@[simp]
theorem Basis.det_comp_basis [Module A M'] (b : Basis ι A M) (b' : Basis ι A M') (f : M →ₗ[A] M') :
    b'.det (f ∘ b) = LinearMap.det (f ∘ₗ (b'.equiv b (Equiv.refl ι) : M' →ₗ[A] M)) := by
  rw [Basis.det_apply, ← LinearMap.det_toMatrix b', LinearMap.toMatrix_comp _ b, Matrix.det_mul,
    LinearMap.toMatrix_basis_equiv, Matrix.det_one, mul_one]
  /-
    M : Type u_2
    inst✝⁶ : AddCommGroup M
    M' : Type u_3
    inst✝⁵ : AddCommGroup M'
    ι : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : Module A M'
    b : Basis ι A M
    b' : Basis ι A M'
    f : LinearMap (RingHom.id A) M M'
    ⊢ Eq (b'.toMatrix (Function.comp ⇑f ⇑b)).det ((LinearMap.toMatrix b b') f).det
  -/
  congr 1; ext i j
  /-
    case e_M.a
    M : Type u_2
    inst✝⁶ : AddCommGroup M
    M' : Type u_3
    inst✝⁵ : AddCommGroup M'
    ι : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    A : Type u_5
    inst✝² : CommRing A
    inst✝¹ : Module A M
    inst✝ : Module A M'
    b : Basis ι A M
    b' : Basis ι A M'
    f : LinearMap (RingHom.id A) M M'
    i j : ι
    ⊢ Eq (b'.toMatrix (Function.comp ⇑f ⇑b) i j) ((LinearMap.toMatrix b b') f i j)
  -/
  rw [Basis.toMatrix_apply, LinearMap.toMatrix_apply, Function.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.det_basis (b : Basis ι A M) (b' : Basis ι A M) :
    LinearMap.det (b'.equiv b (Equiv.refl ι)).toLinearMap = b'.det b :=
  (b.det_comp_basis b' (LinearMap.id)).symm


theorem Basis.det_inv (b : Basis ι A M) (b' : Basis ι A M) :
    (b.isUnit_det b').unit⁻¹ = b'.det b := by
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b b' : Basis ι A M
    ⊢ Eq (↑(Inv.inv ⋯.unit)) (b'.det ⇑b)
  -/
  rw [← Units.mul_eq_one_iff_inv_eq, IsUnit.unit_spec, ← Basis.det_basis, ← Basis.det_basis]
  /-
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    A : Type u_5
    inst✝¹ : CommRing A
    inst✝ : Module A M
    b b' : Basis ι A M
    ⊢ Eq (HMul.hMul (LinearMap.det ↑(b.equiv b' (Equiv.refl ι))) (LinearMap.det ↑( …
  -/
  exact LinearEquiv.det_mul_det_symm _
  /-
    🎉 no goals
  -/


theorem Basis.det_reindex {ι' : Type*} [Fintype ι'] [DecidableEq ι'] (b : Basis ι R M) (v : ι' → M)
    (e : ι ≃ ι') : (b.reindex e).det v = b.det (v ∘ e) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    ι' : Type u_6
    inst✝¹ : Fintype ι'
    inst✝ : DecidableEq ι'
    b : Basis ι R M
    v : ι' → M
    e : Equiv ι ι'
    ⊢ Eq ((b.reindex e).det v) (b.det (Function.comp v ⇑e))
  -/
  rw [Basis.det_apply, Basis.toMatrix_reindex', det_reindexAlgEquiv, Basis.det_apply]
  /-
    🎉 no goals
  -/


theorem Basis.det_reindex' {ι' : Type*} [Fintype ι'] [DecidableEq ι'] (b : Basis ι R M)
    (e : ι ≃ ι') : (b.reindex e).det = b.det.domDomCongr e :=
  AlternatingMap.ext fun _ => Basis.det_reindex _ _ _


theorem Basis.det_reindex_symm {ι' : Type*} [Fintype ι'] [DecidableEq ι'] (b : Basis ι R M)
    (v : ι → M) (e : ι' ≃ ι) : (b.reindex e.symm).det (v ∘ e) = b.det v := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    ι : Type u_4
    inst✝³ : DecidableEq ι
    inst✝² : Fintype ι
    ι' : Type u_6
    inst✝¹ : Fintype ι'
    inst✝ : DecidableEq ι'
    b : Basis ι R M
    v : ι → M
    e : Equiv ι' ι
    ⊢ Eq ((b.reindex e.symm).det (Function.comp v ⇑e)) (b.det v)
  -/
  rw [Basis.det_reindex, Function.comp_assoc, e.self_comp_symm, Function.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.det_map (b : Basis ι R M) (f : M ≃ₗ[R] M') (v : ι → M') :
    (b.map f).det v = b.det (f.symm ∘ v) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    M' : Type u_3
    inst✝³ : AddCommGroup M'
    inst✝² : Module R M'
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    f : LinearEquiv (RingHom.id R) M M'
    v : ι → M'
    ⊢ Eq ((b.map f).det v) (b.det (Function.comp (⇑f.symm) v))
  -/
  rw [Basis.det_apply, Basis.toMatrix_map, Basis.det_apply]
  /-
    🎉 no goals
  -/


theorem Basis.det_map' (b : Basis ι R M) (f : M ≃ₗ[R] M') :
    (b.map f).det = b.det.compLinearMap f.symm :=
  AlternatingMap.ext <| b.det_map f


@[simp]
theorem Pi.basisFun_det : (Pi.basisFun R ι).det = Matrix.detRowAlternating := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    ⊢ Eq (Pi.basisFun R ι).det Matrix.detRowAlternating
  -/
  ext M
  /-
    case H
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    M : ι → ι → R
    ⊢ Eq ((Pi.basisFun R ι).det M) (Matrix.detRowAlternating M)
  -/
  rw [Basis.det_apply, Basis.coePiBasisFun.toMatrix_eq_transpose, det_transpose]
  /-
    🎉 no goals
  -/


theorem Pi.basisFun_det_apply (v : ι → ι → R) :
    (Pi.basisFun R ι).det v = (Matrix.of v).det := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    v : ι → ι → R
    ⊢ Eq ((Pi.basisFun R ι).det v) (Matrix.of v).det
  -/
  rw [Pi.basisFun_det]
  /-
    R : Type u_1
    inst✝² : CommRing R
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    v : ι → ι → R
    ⊢ Eq (Matrix.detRowAlternating v) (Matrix.of v).det
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If we fix a background basis `e`, then for any other basis `v`, we can characterise the
coordinates provided by `v` in terms of determinants relative to `e`. -/
theorem Basis.det_smul_mk_coord_eq_det_update {v : ι → M} (hli : LinearIndependent R v)
    (hsp : ⊤ ≤ span R (range v)) (i : ι) :
    e.det v • (Basis.mk hli hsp).coord i = e.det.toMultilinearMap.toLinearMap v i := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    v : ι → M
    hli : LinearIndependent R v
    hsp : LE.le Top.top (Submodule.span R (Set.range v))
    i : ι
    ⊢ Eq (HSMul.hSMul (e.det v) ((Basis.mk hli hsp).coord i)) ((↑e.det).toLinearMa …
  -/
  apply (Basis.mk hli hsp).ext
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    v : ι → M
    hli : LinearIndependent R v
    hsp : LE.le Top.top (Submodule.span R (Set.range v))
    i : ι
    ⊢ ∀ (i_1 : ι), Eq ((HSMul.hSMul (e.det v) ((Basis.mk hli hsp).coord i)) ((Basi …
  -/
  intro k
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    v : ι → M
    hli : LinearIndependent R v
    hsp : LE.le Top.top (Submodule.span R (Set.range v))
    i k : ι
    ⊢ Eq ((HSMul.hSMul (e.det v) ((Basis.mk hli hsp).coord i)) ((Basis.mk hli hsp) …
  -/
  rcases eq_or_ne k i with (rfl | hik) <;>
    simp only [Algebra.id.smul_eq_mul, Basis.coe_mk, LinearMap.smul_apply, LinearMap.coe_mk,
      MultilinearMap.toLinearMap_apply]
    /-
      case inl
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      k : ι
      ⊢ Eq (HMul.hMul (e.det v) (((Basis.mk hli hsp).coord k) (v k))) (↑e.det (Funct …
    -/
  · rw [Basis.mk_coord_apply_eq, mul_one, update_eq_self]
    /-
      case inl
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      k : ι
      ⊢ Eq (e.det v) (↑e.det v)
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      i k : ι
      hik : Ne k i
      ⊢ Eq (HMul.hMul (e.det v) (((Basis.mk hli hsp).coord i) (v k))) (↑e.det (Funct …
    -/
  · rw [Basis.mk_coord_apply_ne hik, mul_zero, eq_comm]
    /-
      case inr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_4
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      e : Basis ι R M
      v : ι → M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      i k : ι
      hik : Ne k i
      ⊢ Eq (↑e.det (Function.update v i (v k))) 0
    -/
    exact e.det.map_eq_zero_of_eq _ (by simp [hik, Function.update_apply]) hik
    /-
      🎉 no goals
    -/


/-- If a basis is multiplied columnwise by scalars `w : ι → Rˣ`, then the determinant with respect
to this basis is multiplied by the product of the inverse of these scalars. -/
theorem Basis.det_unitsSMul (e : Basis ι R M) (w : ι → Rˣ) :
    (e.unitsSMul w).det = (↑(∏ i, w i)⁻¹ : R) • e.det := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    w : ι → Units R
    ⊢ Eq (e.unitsSMul w).det (HSMul.hSMul (↑(Inv.inv (Finset.univ.prod fun i => w  …
  -/
  ext f
  change
    (Matrix.det fun i j => (e.unitsSMul w).repr (f j) i) =
      (↑(∏ i, w i)⁻¹ : R) • Matrix.det fun i j => e.repr (f j) i
  /-
    case H
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    w : ι → Units R
    f : ι → M
    ⊢ Eq (Matrix.det fun i j => ((e.unitsSMul w).repr (f j)) i) (HSMul.hSMul (↑(In …
  -/
  simp only [e.repr_unitsSMul]
  /-
    case H
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    w : ι → Units R
    f : ι → M
    ⊢ Eq (Matrix.det fun i j => HSMul.hSMul (Inv.inv (w i)) ((e.repr (f j)) i)) (H …
  -/
  convert Matrix.det_mul_column (fun i => (↑(w i)⁻¹ : R)) fun i j => e.repr (f j) i
  /-
    case h.e'_3.h.e'_1
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    w : ι → Units R
    f : ι → M
    ⊢ Eq (↑(Inv.inv (Finset.univ.prod fun i => w i))) (Finset.univ.prod fun i => ↑ …
  -/
  simp [← Finset.prod_inv_distrib]
  /-
    🎉 no goals
  -/


/-- The determinant of a basis constructed by `unitsSMul` is the product of the given units. -/
@[simp]
theorem Basis.det_unitsSMul_self (w : ι → Rˣ) : e.det (e.unitsSMul w) = ∏ i, (w i : R) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    e : Basis ι R M
    w : ι → Units R
    ⊢ Eq (e.det ⇑(e.unitsSMul w)) (Finset.univ.prod fun i => ↑(w i))
  -/
  simp [Basis.det_apply]
  /-
    🎉 no goals
  -/


/-- The determinant of a basis constructed by `isUnitSMul` is the product of the given units. -/
@[simp]
theorem Basis.det_isUnitSMul {w : ι → R} (hw : ∀ i, IsUnit (w i)) :
    e.det (e.isUnitSMul hw) = ∏ i, w i :=
  e.det_unitsSMul_self _

