/-- The change-of-basis matrix between two orthonormal bases with the same orientation has
determinant 1. -/
theorem det_to_matrix_orthonormalBasis_of_same_orientation
    (h : e.toBasis.orientation = f.toBasis.orientation) : e.toBasis.det f = 1 := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e f : OrthonormalBasis ι Real E
    h : Eq e.toBasis.orientation f.toBasis.orientation
    ⊢ Eq (e.toBasis.det ⇑f) 1
  -/
  apply (e.det_to_matrix_orthonormalBasis_real f).resolve_right
  have : 0 < e.toBasis.det f := by
    rw [e.toBasis.orientation_eq_iff_det_pos] at h
    simpa using h
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e f : OrthonormalBasis ι Real E
    h : Eq e.toBasis.orientation f.toBasis.orientation
    this : LT.lt 0 (e.toBasis.det ⇑f)
    ⊢ Not (Eq (e.toBasis.det ⇑f) (-1))
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The change-of-basis matrix between two orthonormal bases with the opposite orientations has
determinant -1. -/
theorem det_to_matrix_orthonormalBasis_of_opposite_orientation
    (h : e.toBasis.orientation ≠ f.toBasis.orientation) : e.toBasis.det f = -1 := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e f : OrthonormalBasis ι Real E
    h : Ne e.toBasis.orientation f.toBasis.orientation
    ⊢ Eq (e.toBasis.det ⇑f) (-1)
  -/
  contrapose! h
  simp [e.toBasis.orientation_eq_iff_det_pos,
    (e.det_to_matrix_orthonormalBasis_real f).resolve_right h]


/-- Two orthonormal bases with the same orientation determine the same "determinant" top-dimensional
form on `E`, and conversely. -/
theorem same_orientation_iff_det_eq_det :
    e.toBasis.det = f.toBasis.det ↔ e.toBasis.orientation = f.toBasis.orientation := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e f : OrthonormalBasis ι Real E
    ⊢ Iff (Eq e.toBasis.det f.toBasis.det) (Eq e.toBasis.orientation f.toBasis.ori …
  -/
  constructor
    /-
      case mp
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      ⊢ Eq e.toBasis.det f.toBasis.det → Eq e.toBasis.orientation f.toBasis.orientat …
    -/
  · intro h
    /-
      case mp
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      h : Eq e.toBasis.det f.toBasis.det
      ⊢ Eq e.toBasis.orientation f.toBasis.orientation
    -/
    dsimp [Basis.orientation]
    /-
      case mp
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      h : Eq e.toBasis.det f.toBasis.det
      ⊢ Eq (rayOfNeZero Real e.toBasis.det ⋯) (rayOfNeZero Real f.toBasis.det ⋯)
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case mpr
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      ⊢ Eq e.toBasis.orientation f.toBasis.orientation → Eq e.toBasis.det f.toBasis. …
    -/
  · intro h
    /-
      case mpr
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      h : Eq e.toBasis.orientation f.toBasis.orientation
      ⊢ Eq e.toBasis.det f.toBasis.det
    -/
    rw [e.toBasis.det.eq_smul_basis_det f.toBasis]
    /-
      case mpr
      E : Type u_1
      inst✝³ : NormedAddCommGroup E
      inst✝² : InnerProductSpace Real E
      ι : Type u_2
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      e f : OrthonormalBasis ι Real E
      h : Eq e.toBasis.orientation f.toBasis.orientation
      ⊢ Eq (HSMul.hSMul (e.toBasis.det ⇑f.toBasis) f.toBasis.det) f.toBasis.det
    -/
    simp [e.det_to_matrix_orthonormalBasis_of_same_orientation f h]
    /-
      🎉 no goals
    -/


/-- Two orthonormal bases with opposite orientations determine opposite "determinant"
top-dimensional forms on `E`. -/
theorem det_eq_neg_det_of_opposite_orientation (h : e.toBasis.orientation ≠ f.toBasis.orientation) :
    e.toBasis.det = -f.toBasis.det := by
  /-
    E : Type u_1
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace Real E
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e f : OrthonormalBasis ι Real E
    h : Ne e.toBasis.orientation f.toBasis.orientation
    ⊢ Eq e.toBasis.det (Neg.neg f.toBasis.det)
  -/
  rw [e.toBasis.det.eq_smul_basis_det f.toBasis]
  -- Porting note: added `neg_one_smul` with explicit type
  simp [e.det_to_matrix_orthonormalBasis_of_opposite_orientation f h,
    neg_one_smul ℝ (M := E [⋀^ι]→ₗ[ℝ] ℝ)]


/-- `OrthonormalBasis.adjustToOrientation`, applied to an orthonormal basis, preserves the
property of orthonormality. -/
theorem orthonormal_adjustToOrientation : Orthonormal ℝ (e.toBasis.adjustToOrientation x) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    ⊢ Orthonormal Real ⇑(e.toBasis.adjustToOrientation x)
  -/
  apply e.orthonormal.orthonormal_of_forall_eq_or_eq_neg
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    ⊢ ∀ (i : ι), Or (Eq ((e.toBasis.adjustToOrientation x) i) (e i)) (Eq ((e.toBas …
  -/
  simpa using e.toBasis.adjustToOrientation_apply_eq_or_eq_neg x
  /-
    🎉 no goals
  -/


/-- Given an orthonormal basis and an orientation, return an orthonormal basis giving that
orientation: either the original basis, or one constructed by negating a single (arbitrary) basis
vector. -/
def adjustToOrientation : OrthonormalBasis ι ℝ E :=
  (e.toBasis.adjustToOrientation x).toOrthonormalBasis (e.orthonormal_adjustToOrientation x)


theorem toBasis_adjustToOrientation :
    (e.adjustToOrientation x).toBasis = e.toBasis.adjustToOrientation x :=
  (e.toBasis.adjustToOrientation x).toBasis_toOrthonormalBasis _


/-- `adjustToOrientation` gives an orthonormal basis with the required orientation. -/
@[simp]
theorem orientation_adjustToOrientation : (e.adjustToOrientation x).toBasis.orientation = x := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    ⊢ Eq (e.adjustToOrientation x).toBasis.orientation x
  -/
  rw [e.toBasis_adjustToOrientation]
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    ⊢ Eq (e.toBasis.adjustToOrientation x).orientation x
  -/
  exact e.toBasis.orientation_adjustToOrientation x
  /-
    🎉 no goals
  -/


/-- Every basis vector from `adjustToOrientation` is either that from the original basis or its
negation. -/
theorem adjustToOrientation_apply_eq_or_eq_neg (i : ι) :
    e.adjustToOrientation x i = e i ∨ e.adjustToOrientation x i = -e i := by
  simpa [← e.toBasis_adjustToOrientation] using
    e.toBasis.adjustToOrientation_apply_eq_or_eq_neg x i


theorem det_adjustToOrientation :
    (e.adjustToOrientation x).toBasis.det = e.toBasis.det ∨
      (e.adjustToOrientation x).toBasis.det = -e.toBasis.det := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    ⊢ Or (Eq (e.adjustToOrientation x).toBasis.det e.toBasis.det) (Eq (e.adjustToO …
  -/
  simpa using e.toBasis.det_adjustToOrientation x
  /-
    🎉 no goals
  -/


theorem abs_det_adjustToOrientation (v : ι → E) :
    |(e.adjustToOrientation x).toBasis.det v| = |e.toBasis.det v| := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    ι : Type u_2
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    e : OrthonormalBasis ι Real E
    x : Orientation Real E ι
    inst✝ : Nonempty ι
    v : ι → E
    ⊢ Eq (abs ((e.adjustToOrientation x).toBasis.det v)) (abs (e.toBasis.det v))
  -/
  simp [toBasis_adjustToOrientation]
  /-
    🎉 no goals
  -/


/-- An orthonormal basis, indexed by `Fin n`, with the given orientation. -/
protected def finOrthonormalBasis (hn : 0 < n) (h : finrank ℝ E = n) (x : Orientation ℝ E (Fin n)) :
    OrthonormalBasis (Fin n) ℝ E := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    hn : LT.lt 0 n
    h : Eq (Module.finrank Real E) n
    x : Orientation Real E (Fin n)
    ⊢ OrthonormalBasis (Fin n) Real E
  -/
  haveI := Fin.pos_iff_nonempty.1 hn
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    hn : LT.lt 0 n
    h : Eq (Module.finrank Real E) n
    x : Orientation Real E (Fin n)
    this : Nonempty (Fin n)
    ⊢ OrthonormalBasis (Fin n) Real E
  -/
  haveI : FiniteDimensional ℝ E := .of_finrank_pos <| h.symm ▸ hn
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    hn : LT.lt 0 n
    h : Eq (Module.finrank Real E) n
    x : Orientation Real E (Fin n)
    this✝ : Nonempty (Fin n)
    this : FiniteDimensional Real E
    ⊢ OrthonormalBasis (Fin n) Real E
  -/
  exact ((@stdOrthonormalBasis _ _ _ _ _ this).reindex <| finCongr h).adjustToOrientation x
  /-
    🎉 no goals
  -/


/-- `Orientation.finOrthonormalBasis` gives a basis with the required orientation. -/
@[simp]
theorem finOrthonormalBasis_orientation (hn : 0 < n) (h : finrank ℝ E = n)
    (x : Orientation ℝ E (Fin n)) : (x.finOrthonormalBasis hn h).toBasis.orientation = x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    hn : LT.lt 0 n
    h : Eq (Module.finrank Real E) n
    x : Orientation Real E (Fin n)
    ⊢ Eq (Orientation.finOrthonormalBasis hn h x).toBasis.orientation x
  -/
  haveI := Fin.pos_iff_nonempty.1 hn
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    hn : LT.lt 0 n
    h : Eq (Module.finrank Real E) n
    x : Orientation Real E (Fin n)
    this : Nonempty (Fin n)
    ⊢ Eq (Orientation.finOrthonormalBasis hn h x).toBasis.orientation x
  -/
  haveI : FiniteDimensional ℝ E := .of_finrank_pos <| h.symm ▸ hn
  exact ((@stdOrthonormalBasis _ _ _ _ _ this).reindex <|
    finCongr h).orientation_adjustToOrientation x


/-- The volume form on an oriented real inner product space, a nonvanishing top-dimensional
alternating form uniquely defined by compatibility with the orientation and inner product structure.
-/
irreducible_def volumeForm : E [⋀^Fin n]→ₗ[ℝ] ℝ := by
  classical
    cases' n with n
    · let opos : E [⋀^Fin 0]→ₗ[ℝ] ℝ := .constOfIsEmpty ℝ E (Fin 0) (1 : ℝ)
      exact o.eq_or_eq_neg_of_isEmpty.by_cases (fun _ => opos) fun _ => -opos
    · exact (o.finOrthonormalBasis n.succ_pos _i.out).toBasis.det


@[simp]
theorem volumeForm_zero_pos [_i : Fact (finrank ℝ E = 0)] :
    Orientation.volumeForm (positiveOrientation : Orientation ℝ E (Fin 0)) =
      AlternatingMap.constLinearEquivOfIsEmpty 1 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Eq Module.Oriented.positiveOrientation.volumeForm (AlternatingMap.constLinea …
  -/
  simp [volumeForm, Or.by_cases, if_pos]
  /-
    🎉 no goals
  -/


theorem volumeForm_zero_neg [_i : Fact (finrank ℝ E = 0)] :
    Orientation.volumeForm (-positiveOrientation : Orientation ℝ E (Fin 0)) =
      -AlternatingMap.constLinearEquivOfIsEmpty 1 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Eq (Neg.neg Module.Oriented.positiveOrientation).volumeForm (Neg.neg (Altern …
  -/
  simp_rw [volumeForm, Or.by_cases, positiveOrientation]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Eq (Nat.casesAuxOn (motive := fun a => Eq 0 a → AlternatingMap Real E Real ( …
  -/
  apply if_neg
  /-
    case hnc
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Not (Eq (Neg.neg (rayOfNeZero Real (AlternatingMap.constLinearEquivOfIsEmpty …
  -/
  simp only [neg_rayOfNeZero]
  /-
    case hnc
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Not (Eq (rayOfNeZero Real (Neg.neg (AlternatingMap.constLinearEquivOfIsEmpty …
  -/
  rw [ray_eq_iff, SameRay.sameRay_comm]
  /-
    case hnc
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    _i : Fact (Eq (Module.finrank Real E) 0)
    ⊢ Not (SameRay Real (AlternatingMap.constLinearEquivOfIsEmpty 1) (Neg.neg (Alt …
  -/
  intro h
  simpa using
    congr_arg AlternatingMap.constLinearEquivOfIsEmpty.symm (eq_zero_of_sameRay_self_neg h)


/-- The volume form on an oriented real inner product space can be evaluated as the determinant with
respect to any orthonormal basis of the space compatible with the orientation. -/
theorem volumeForm_robust (b : OrthonormalBasis (Fin n) ℝ E) (hb : b.toBasis.orientation = o) :
    o.volumeForm = b.toBasis.det := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    b : OrthonormalBasis (Fin n) Real E
    hb : Eq b.toBasis.orientation o
    ⊢ Eq o.volumeForm b.toBasis.det
  -/
  cases n
  · classical
      have : o = positiveOrientation := hb.symm.trans b.toBasis.orientation_isEmpty
      simp_rw [volumeForm, Or.by_cases, dif_pos this, Nat.rec_zero, Basis.det_isEmpty]
    /-
      case succ
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n✝ : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n✝ 1))
      o : Orientation Real E (Fin (HAdd.hAdd n✝ 1))
      b : OrthonormalBasis (Fin (HAdd.hAdd n✝ 1)) Real E
      hb : Eq b.toBasis.orientation o
      ⊢ Eq o.volumeForm b.toBasis.det
    -/
  · simp_rw [volumeForm]
    /-
      case succ
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n✝ : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n✝ 1))
      o : Orientation Real E (Fin (HAdd.hAdd n✝ 1))
      b : OrthonormalBasis (Fin (HAdd.hAdd n✝ 1)) Real E
      hb : Eq b.toBasis.orientation o
      ⊢ Eq (Nat.casesAuxOn (motive := fun a => Eq (HAdd.hAdd n✝ 1) a → AlternatingMa …
    -/
    rw [same_orientation_iff_det_eq_det, hb]
    /-
      case succ
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n✝ : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n✝ 1))
      o : Orientation Real E (Fin (HAdd.hAdd n✝ 1))
      b : OrthonormalBasis (Fin (HAdd.hAdd n✝ 1)) Real E
      hb : Eq b.toBasis.orientation o
      ⊢ Eq (Orientation.finOrthonormalBasis ⋯ ⋯ o).toBasis.orientation o
    -/
    exact o.finOrthonormalBasis_orientation _ _
    /-
      🎉 no goals
    -/


/-- The volume form on an oriented real inner product space can be evaluated as the determinant with
respect to any orthonormal basis of the space compatible with the orientation. -/
theorem volumeForm_robust_neg (b : OrthonormalBasis (Fin n) ℝ E) (hb : b.toBasis.orientation ≠ o) :
    o.volumeForm = -b.toBasis.det := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    b : OrthonormalBasis (Fin n) Real E
    hb : Ne b.toBasis.orientation o
    ⊢ Eq o.volumeForm (Neg.neg b.toBasis.det)
  -/
  cases' n with n
  · classical
      have : positiveOrientation ≠ o := by rwa [b.toBasis.orientation_isEmpty] at hb
      simp_rw [volumeForm, Or.by_cases, dif_neg this.symm, Nat.rec_zero, Basis.det_isEmpty]
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    b : OrthonormalBasis (Fin (HAdd.hAdd n 1)) Real E
    hb : Ne b.toBasis.orientation o
    ⊢ Eq o.volumeForm (Neg.neg b.toBasis.det)
  -/
  let e : OrthonormalBasis (Fin n.succ) ℝ E := o.finOrthonormalBasis n.succ_pos Fact.out
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    b : OrthonormalBasis (Fin (HAdd.hAdd n 1)) Real E
    hb : Ne b.toBasis.orientation o
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    ⊢ Eq o.volumeForm (Neg.neg b.toBasis.det)
  -/
  simp_rw [volumeForm]
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    b : OrthonormalBasis (Fin (HAdd.hAdd n 1)) Real E
    hb : Ne b.toBasis.orientation o
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    ⊢ Eq (Nat.casesAuxOn (motive := fun a => Eq (HAdd.hAdd n 1) a → AlternatingMap …
  -/
  apply e.det_eq_neg_det_of_opposite_orientation b
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    b : OrthonormalBasis (Fin (HAdd.hAdd n 1)) Real E
    hb : Ne b.toBasis.orientation o
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    ⊢ Ne e.toBasis.orientation b.toBasis.orientation
  -/
  convert hb.symm
  /-
    case h.e'_2
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    b : OrthonormalBasis (Fin (HAdd.hAdd n 1)) Real E
    hb : Ne b.toBasis.orientation o
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    ⊢ Eq e.toBasis.orientation o
  -/
  exact o.finOrthonormalBasis_orientation _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem volumeForm_neg_orientation : (-o).volumeForm = -o.volumeForm := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    ⊢ Eq (Neg.neg o).volumeForm (Neg.neg o.volumeForm)
  -/
  cases' n with n
    /-
      case zero
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      ⊢ Eq (Neg.neg o).volumeForm (Neg.neg o.volumeForm)
    -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl
      /-
        case zero.refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        _i : Fact (Eq (Module.finrank Real E) 0)
        ⊢ Eq (Neg.neg Module.Oriented.positiveOrientation).volumeForm (Neg.neg Module. …
      -/
    · simp [volumeForm_zero_neg]
      /-
        🎉 no goals
      -/
      /-
        case zero.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        _i : Fact (Eq (Module.finrank Real E) 0)
        ⊢ Eq (Neg.neg (Neg.neg Module.Oriented.positiveOrientation)).volumeForm (Neg.n …
      -/
    · rw [neg_neg (positiveOrientation (R := ℝ))] -- Porting note: added
      /-
        case zero.refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        _i : Fact (Eq (Module.finrank Real E) 0)
        ⊢ Eq Module.Oriented.positiveOrientation.volumeForm (Neg.neg (Neg.neg Module.O …
      -/
      simp [volumeForm_zero_neg]
      /-
        🎉 no goals
      -/
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    ⊢ Eq (Neg.neg o).volumeForm (Neg.neg o.volumeForm)
  -/
  let e : OrthonormalBasis (Fin n.succ) ℝ E := o.finOrthonormalBasis n.succ_pos Fact.out
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    ⊢ Eq (Neg.neg o).volumeForm (Neg.neg o.volumeForm)
  -/
  have h₁ : e.toBasis.orientation = o := o.finOrthonormalBasis_orientation _ _
  have h₂ : e.toBasis.orientation ≠ -o := by
    symm
    rw [e.toBasis.orientation_ne_iff_eq_neg, h₁]
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    h₁ : Eq e.toBasis.orientation o
    h₂ : Ne e.toBasis.orientation (Neg.neg o)
    ⊢ Eq (Neg.neg o).volumeForm (Neg.neg o.volumeForm)
  -/
  rw [o.volumeForm_robust e h₁, (-o).volumeForm_robust_neg e h₂]
  /-
    🎉 no goals
  -/


theorem volumeForm_robust' (b : OrthonormalBasis (Fin n) ℝ E) (v : Fin n → E) :
    |o.volumeForm v| = |b.toBasis.det v| := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    b : OrthonormalBasis (Fin n) Real E
    v : Fin n → E
    ⊢ Eq (abs (o.volumeForm v)) (abs (b.toBasis.det v))
  -/
  cases n
    /-
      case zero
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      b : OrthonormalBasis (Fin 0) Real E
      v : Fin 0 → E
      ⊢ Eq (abs (o.volumeForm v)) (abs (b.toBasis.det v))
    -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · rw [o.volumeForm_robust (b.adjustToOrientation o) (b.orientation_adjustToOrientation o),
      b.abs_det_adjustToOrientation]


/-- Let `v` be an indexed family of `n` vectors in an oriented `n`-dimensional real inner
product space `E`. The output of the volume form of `E` when evaluated on `v` is bounded in absolute
value by the product of the norms of the vectors `v i`. -/
theorem abs_volumeForm_apply_le (v : Fin n → E) : |o.volumeForm v| ≤ ∏ i : Fin n, ‖v i‖ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    v : Fin n → E
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  cases' n with n
    /-
      case zero
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      v : Fin 0 → E
      ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
    -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  haveI : FiniteDimensional ℝ E := .of_fact_finrank_eq_succ n
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this : FiniteDimensional Real E
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  have : finrank ℝ E = Fintype.card (Fin n.succ) := by simpa using _i.out
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  let b : OrthonormalBasis (Fin n.succ) ℝ E := gramSchmidtOrthonormalBasis this v
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  have hb : b.toBasis.det v = ∏ i, ⟪b i, v i⟫ := gramSchmidtOrthonormalBasis_det this v
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    ⊢ LE.le (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  rw [o.volumeForm_robust' b, hb, Finset.abs_prod]
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    ⊢ LE.le (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ …
  -/
  apply Finset.prod_le_prod
    /-
      case succ.h0
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      this✝ : FiniteDimensional Real E
      this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
      hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      ⊢ ∀ (i : Fin n.succ), Membership.mem Finset.univ i → LE.le 0 (abs (Inner.inner …
    -/
  · intro i _
    /-
      case succ.h0
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      this✝ : FiniteDimensional Real E
      this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
      hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      i : Fin n.succ
      a✝ : Membership.mem Finset.univ i
      ⊢ LE.le 0 (abs (Inner.inner (b i) (v i)))
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case succ.h1
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    ⊢ ∀ (i : Fin n.succ), Membership.mem Finset.univ i → LE.le (abs (Inner.inner ( …
  -/
  intro i _
  /-
    case succ.h1
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    i : Fin n.succ
    a✝ : Membership.mem Finset.univ i
    ⊢ LE.le (abs (Inner.inner (b i) (v i))) (Norm.norm (v i))
  -/
  convert abs_real_inner_le_norm (b i) (v i)
  /-
    case h.e'_4
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    this✝ : FiniteDimensional Real E
    this : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis this v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    i : Fin n.succ
    a✝ : Membership.mem Finset.univ i
    ⊢ Eq (Norm.norm (v i)) (HMul.hMul (Norm.norm (b i)) (Norm.norm (v i)))
  -/
  simp [b.orthonormal.1 i]
  /-
    🎉 no goals
  -/


theorem volumeForm_apply_le (v : Fin n → E) : o.volumeForm v ≤ ∏ i : Fin n, ‖v i‖ :=
  (le_abs_self _).trans (o.abs_volumeForm_apply_le v)


/-- Let `v` be an indexed family of `n` orthogonal vectors in an oriented `n`-dimensional
real inner product space `E`. The output of the volume form of `E` when evaluated on `v` is, up to
sign, the product of the norms of the vectors `v i`. -/
theorem abs_volumeForm_apply_of_pairwise_orthogonal {v : Fin n → E}
    (hv : Pairwise fun i j => ⟪v i, v j⟫ = 0) : |o.volumeForm v| = ∏ i : Fin n, ‖v i‖ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    v : Fin n → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  cases' n with n
    /-
      case zero
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      v : Fin 0 → E
      hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
    -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  haveI : FiniteDimensional ℝ E := .of_fact_finrank_eq_succ n
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  have hdim : finrank ℝ E = Fintype.card (Fin n.succ) := by simpa using _i.out
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  let b : OrthonormalBasis (Fin n.succ) ℝ E := gramSchmidtOrthonormalBasis hdim v
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  have hb : b.toBasis.det v = ∏ i, ⟪b i, v i⟫ := gramSchmidtOrthonormalBasis_det hdim v
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    ⊢ Eq (abs (o.volumeForm v)) (Finset.univ.prod fun i => Norm.norm (v i))
  -/
  rw [o.volumeForm_robust' b, hb, Finset.abs_prod]
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    ⊢ Eq (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ.pr …
  -/
  by_cases h : ∃ i, v i = 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      this : FiniteDimensional Real E
      hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
      hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      h : Exists fun i => Eq (v i) 0
      ⊢ Eq (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ.pr …
    -/
  · obtain ⟨i, hi⟩ := h
    /-
      case pos.intro
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      this : FiniteDimensional Real E
      hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
      hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (v i) 0
      ⊢ Eq (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ.pr …
    -/
    rw [Finset.prod_eq_zero (Finset.mem_univ i), Finset.prod_eq_zero (Finset.mem_univ i)] <;>
      /-
        case pos.intro
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace Real E
        n : Nat
        _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
        o : Orientation Real E (Fin (HAdd.hAdd n 1))
        v : Fin (HAdd.hAdd n 1) → E
        hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
        this : FiniteDimensional Real E
        hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
        b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
        hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
        i : Fin (HAdd.hAdd n 1)
        hi : Eq (v i) 0
        ⊢ Eq (Norm.norm (v i)) 0
      -/
      /-
        🎉 no goals
      -/
      simp [hi]
      /-
        🎉 no goals
      -/
  /-
    case neg
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : Not (Exists fun i => Eq (v i) 0)
    ⊢ Eq (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ.pr …
  -/
  push_neg at h
  /-
    case neg
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
    ⊢ Eq (Finset.univ.prod fun x => abs (Inner.inner (b x) (v x))) (Finset.univ.pr …
  -/
  congr
  /-
    case neg.e_f
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
    ⊢ Eq (fun x => abs (Inner.inner (b x) (v x))) fun i => Norm.norm (v i)
  -/
  ext i
  /-
    case neg.e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
    i : Fin n.succ
    ⊢ Eq (abs (Inner.inner (b i) (v i))) (Norm.norm (v i))
  -/
  have hb : b i = ‖v i‖⁻¹ • v i := gramSchmidtOrthonormalBasis_apply_of_orthogonal hdim hv (h i)
  /-
    case neg.e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb✝ : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
    i : Fin n.succ
    hb : Eq (b i) (HSMul.hSMul (Inv.inv (Norm.norm (v i))) (v i))
    ⊢ Eq (abs (Inner.inner (b i) (v i))) (Norm.norm (v i))
  -/
  simp only [hb, inner_smul_left, real_inner_self_eq_norm_mul_norm, RCLike.conj_to_real]
  /-
    case neg.e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    v : Fin (HAdd.hAdd n 1) → E
    hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    this : FiniteDimensional Real E
    hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
    b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
    hb✝ : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
    h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
    i : Fin n.succ
    hb : Eq (b i) (HSMul.hSMul (Inv.inv (Norm.norm (v i))) (v i))
    ⊢ Eq (abs (HMul.hMul (Inv.inv (Norm.norm (v i))) (HMul.hMul (Norm.norm (v i))  …
  -/
  rw [abs_of_nonneg]
    /-
      case neg.e_f.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      this : FiniteDimensional Real E
      hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
      hb✝ : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
      i : Fin n.succ
      hb : Eq (b i) (HSMul.hSMul (Inv.inv (Norm.norm (v i))) (v i))
      ⊢ Eq (HMul.hMul (Inv.inv (Norm.norm (v i))) (HMul.hMul (Norm.norm (v i)) (Norm …
    -/
  · field_simp
    /-
      🎉 no goals
    -/
    /-
      case neg.e_f.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      v : Fin (HAdd.hAdd n 1) → E
      hv : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
      this : FiniteDimensional Real E
      hdim : Eq (Module.finrank Real E) (Fintype.card (Fin n.succ))
      b : OrthonormalBasis (Fin n.succ) Real E := gramSchmidtOrthonormalBasis hdim v
      hb✝ : Eq (b.toBasis.det v) (Finset.univ.prod fun i => Inner.inner (b i) (v i))
      h : ∀ (i : Fin (HAdd.hAdd n 1)), Ne (v i) 0
      i : Fin n.succ
      hb : Eq (b i) (HSMul.hSMul (Inv.inv (Norm.norm (v i))) (v i))
      ⊢ LE.le 0 (HMul.hMul (Inv.inv (Norm.norm (v i))) (HMul.hMul (Norm.norm (v i))  …
    -/
  · positivity
    /-
      🎉 no goals
    -/


/-- The output of the volume form of an oriented real inner product space `E` when evaluated on an
orthonormal basis is ±1. -/
theorem abs_volumeForm_apply_of_orthonormal (v : OrthonormalBasis (Fin n) ℝ E) :
    |o.volumeForm v| = 1 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    v : OrthonormalBasis (Fin n) Real E
    ⊢ Eq (abs (o.volumeForm ⇑v)) 1
  -/
  simpa [o.volumeForm_robust' v v] using congr_arg abs v.toBasis.det_self
  /-
    🎉 no goals
  -/


theorem volumeForm_map {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [Fact (finrank ℝ F = n)] (φ : E ≃ₗᵢ[ℝ] F) (x : Fin n → F) :
    (Orientation.map (Fin n) φ.toLinearEquiv o).volumeForm x = o.volumeForm (φ.symm ∘ x) := by
  /-
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    inst✝ : Fact (Eq (Module.finrank Real F) n)
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    x : Fin n → F
    ⊢ Eq (((Orientation.map (Fin n) φ.toLinearEquiv) o).volumeForm x) (o.volumeFor …
  -/
  cases' n with n
    /-
      case zero
      E : Type u_1
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace Real E
      F : Type u_2
      inst✝² : NormedAddCommGroup F
      inst✝¹ : InnerProductSpace Real F
      φ : LinearIsometryEquiv (RingHom.id Real) E F
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      inst✝ : Fact (Eq (Module.finrank Real F) 0)
      x : Fin 0 → F
      ⊢ Eq (((Orientation.map (Fin 0) φ.toLinearEquiv) o).volumeForm x) (o.volumeFor …
    -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    case succ
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    inst✝ : Fact (Eq (Module.finrank Real F) (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → F
    ⊢ Eq (((Orientation.map (Fin (HAdd.hAdd n 1)) φ.toLinearEquiv) o).volumeForm x …
  -/
  let e : OrthonormalBasis (Fin n.succ) ℝ E := o.finOrthonormalBasis n.succ_pos Fact.out
  have he : e.toBasis.orientation = o :=
    o.finOrthonormalBasis_orientation n.succ_pos Fact.out
  have heφ : (e.map φ).toBasis.orientation = Orientation.map (Fin n.succ) φ.toLinearEquiv o := by
    rw [← he]
    exact e.toBasis.orientation_map φ.toLinearEquiv
  /-
    case succ
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    inst✝ : Fact (Eq (Module.finrank Real F) (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → F
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    he : Eq e.toBasis.orientation o
    heφ : Eq (e.map φ).toBasis.orientation ((Orientation.map (Fin n.succ) φ.toLine …
    ⊢ Eq (((Orientation.map (Fin (HAdd.hAdd n 1)) φ.toLinearEquiv) o).volumeForm x …
  -/
  rw [(Orientation.map (Fin n.succ) φ.toLinearEquiv o).volumeForm_robust (e.map φ) heφ]
  /-
    case succ
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    inst✝ : Fact (Eq (Module.finrank Real F) (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → F
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    he : Eq e.toBasis.orientation o
    heφ : Eq (e.map φ).toBasis.orientation ((Orientation.map (Fin n.succ) φ.toLine …
    ⊢ Eq ((e.map φ).toBasis.det x) (o.volumeForm (Function.comp (⇑φ.symm) x))
  -/
  rw [o.volumeForm_robust e he]
  /-
    case succ
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real F
    φ : LinearIsometryEquiv (RingHom.id Real) E F
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    inst✝ : Fact (Eq (Module.finrank Real F) (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → F
    e : OrthonormalBasis (Fin n.succ) Real E := Orientation.finOrthonormalBasis ⋯  …
    he : Eq e.toBasis.orientation o
    heφ : Eq (e.map φ).toBasis.orientation ((Orientation.map (Fin n.succ) φ.toLine …
    ⊢ Eq ((e.map φ).toBasis.det x) (e.toBasis.det (Function.comp (⇑φ.symm) x))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The volume form is invariant under pullback by a positively-oriented isometric automorphism. -/
theorem volumeForm_comp_linearIsometryEquiv (φ : E ≃ₗᵢ[ℝ] E)
    (hφ : 0 < LinearMap.det (φ.toLinearEquiv : E →ₗ[ℝ] E)) (x : Fin n → E) :
    o.volumeForm (φ ∘ x) = o.volumeForm x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) n)
    o : Orientation Real E (Fin n)
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    x : Fin n → E
    ⊢ Eq (o.volumeForm (Function.comp (⇑φ) x)) (o.volumeForm x)
  -/
  cases' n with n -- Porting note: need to explicitly prove `FiniteDimensional ℝ E`
    /-
      case zero
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      _i : Fact (Eq (Module.finrank Real E) 0)
      o : Orientation Real E (Fin 0)
      x : Fin 0 → E
      ⊢ Eq (o.volumeForm (Function.comp (⇑φ) x)) (o.volumeForm x)
    -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  · refine o.eq_or_eq_neg_of_isEmpty.elim ?_ ?_ <;> rintro rfl <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → E
    ⊢ Eq (o.volumeForm (Function.comp (⇑φ) x)) (o.volumeForm x)
  -/
  haveI : FiniteDimensional ℝ E := .of_fact_finrank_eq_succ n
  /-
    case succ
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    φ : LinearIsometryEquiv (RingHom.id Real) E E
    hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
    n : Nat
    _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
    o : Orientation Real E (Fin (HAdd.hAdd n 1))
    x : Fin (HAdd.hAdd n 1) → E
    this : FiniteDimensional Real E
    ⊢ Eq (o.volumeForm (Function.comp (⇑φ) x)) (o.volumeForm x)
  -/
  convert o.volumeForm_map φ (φ ∘ x)
    /-
      case h.e'_2.h.e'_5.h.e'_6
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd n 1) → E
      this : FiniteDimensional Real E
      ⊢ Eq o ((Orientation.map (Fin (HAdd.hAdd n 1)) φ.toLinearEquiv) o)
    -/
  · symm
    /-
      case h.e'_2.h.e'_5.h.e'_6
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd n 1) → E
      this : FiniteDimensional Real E
      ⊢ Eq ((Orientation.map (Fin (HAdd.hAdd n 1)) φ.toLinearEquiv) o) o
    -/
    rwa [← o.map_eq_iff_det_pos φ.toLinearEquiv] at hφ
    /-
      case h.e'_2.h.e'_5.h.e'_6
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd n 1) → E
      this : FiniteDimensional Real E
      ⊢ Eq (Fintype.card (Fin (HAdd.hAdd n 1))) (Module.finrank Real E)
    -/
    rw [_i.out, Fintype.card_fin]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd n 1) → E
      this : FiniteDimensional Real E
      ⊢ Eq x (Function.comp (⇑φ.symm) (Function.comp (⇑φ) x))
    -/
  · ext
    /-
      case h.e'_3.h.e'_6.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace Real E
      φ : LinearIsometryEquiv (RingHom.id Real) E E
      hφ : LT.lt 0 (LinearMap.det ↑φ.toLinearEquiv)
      n : Nat
      _i : Fact (Eq (Module.finrank Real E) (HAdd.hAdd n 1))
      o : Orientation Real E (Fin (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd n 1) → E
      this : FiniteDimensional Real E
      x✝ : Fin (HAdd.hAdd n 1)
      ⊢ Eq (x x✝) (Function.comp (⇑φ.symm) (Function.comp (⇑φ) x) x✝)
    -/
    simp
    /-
      🎉 no goals
    -/


