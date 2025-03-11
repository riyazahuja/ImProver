/-- A `NormedAddTorsor V P` is a torsor of an additive seminormed group
action by a `SeminormedAddCommGroup V` on points `P`. We bundle the pseudometric space
structure and require the distance to be the same as results from the
norm (which in fact implies the distance yields a pseudometric space, but
bundling just the distance and using an instance for the pseudometric space
results in type class problems). -/
class NormedAddTorsor (V : outParam Type*) (P : Type*) [SeminormedAddCommGroup V]
  [PseudoMetricSpace P] extends AddTorsor V P where
  dist_eq_norm' : ∀ x y : P, dist x y = ‖(x -ᵥ y : V)‖


/-- Shortcut instance to help typeclass inference out. -/
instance (priority := 100) NormedAddTorsor.toAddTorsor' {V P : Type*} [NormedAddCommGroup V]
    [MetricSpace P] [NormedAddTorsor V P] : AddTorsor V P :=
  NormedAddTorsor.toAddTorsor


instance (priority := 100) NormedAddTorsor.to_isometricVAdd : IsometricVAdd V P :=
  ⟨fun c => Isometry.of_dist_eq fun x y => by
    /-
      α : Type u_1
      V : Type u_2
      P : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁵ : SeminormedAddCommGroup V
      inst✝⁴ : PseudoMetricSpace P
      inst✝³ : NormedAddTorsor V P
      inst✝² : NormedAddCommGroup W
      inst✝¹ : MetricSpace Q
      inst✝ : NormedAddTorsor W Q
      c : V
      x y : P
      ⊢ Eq (Dist.dist (HVAdd.hVAdd c x) (HVAdd.hVAdd c y)) (Dist.dist x y)
    -/
    simp [NormedAddTorsor.dist_eq_norm']⟩
    /-
      🎉 no goals
    -/


/-- A `SeminormedAddCommGroup` is a `NormedAddTorsor` over itself. -/
instance (priority := 100) SeminormedAddCommGroup.toNormedAddTorsor : NormedAddTorsor V V where
  dist_eq_norm' := dist_eq_norm

-- Because of the AddTorsor.nonempty instance.

/-- A nonempty affine subspace of a `NormedAddTorsor` is itself a `NormedAddTorsor`. -/
instance AffineSubspace.toNormedAddTorsor {R : Type*} [Ring R] [Module R V]
    (s : AffineSubspace R P) [Nonempty s] : NormedAddTorsor s.direction s :=
  { AffineSubspace.toAddTorsor s with
    dist_eq_norm' := fun x y => NormedAddTorsor.dist_eq_norm' x.val y.val }


/-- The distance equals the norm of subtracting two points. In this
lemma, it is necessary to have `V` as an explicit argument; otherwise
`rw dist_eq_norm_vsub` sometimes doesn't work. -/
theorem dist_eq_norm_vsub (x y : P) : dist x y = ‖x -ᵥ y‖ :=
  NormedAddTorsor.dist_eq_norm' x y


theorem nndist_eq_nnnorm_vsub (x y : P) : nndist x y = ‖x -ᵥ y‖₊ :=
  NNReal.eq <| dist_eq_norm_vsub V x y



/-- The distance equals the norm of subtracting two points. In this
lemma, it is necessary to have `V` as an explicit argument; otherwise
`rw dist_eq_norm_vsub'` sometimes doesn't work. -/
theorem dist_eq_norm_vsub' (x y : P) : dist x y = ‖y -ᵥ x‖ :=
  (dist_comm _ _).trans (dist_eq_norm_vsub _ _ _)


theorem nndist_eq_nnnorm_vsub' (x y : P) : nndist x y = ‖y -ᵥ x‖₊ :=
  NNReal.eq <| dist_eq_norm_vsub' V x y


theorem dist_vadd_cancel_left (v : V) (x y : P) : dist (v +ᵥ x) (v +ᵥ y) = dist x y :=
  dist_vadd _ _ _


theorem nndist_vadd_cancel_left (v : V) (x y : P) : nndist (v +ᵥ x) (v +ᵥ y) = nndist x y :=
  NNReal.eq <| dist_vadd_cancel_left _ _ _


@[simp]
theorem dist_vadd_cancel_right (v₁ v₂ : V) (x : P) : dist (v₁ +ᵥ x) (v₂ +ᵥ x) = dist v₁ v₂ := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v₁ v₂ : V
    x : P
    ⊢ Eq (Dist.dist (HVAdd.hVAdd v₁ x) (HVAdd.hVAdd v₂ x)) (Dist.dist v₁ v₂)
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm, vadd_vsub_vadd_cancel_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_vadd_cancel_right (v₁ v₂ : V) (x : P) : nndist (v₁ +ᵥ x) (v₂ +ᵥ x) = nndist v₁ v₂ :=
  NNReal.eq <| dist_vadd_cancel_right _ _ _


@[simp]
theorem dist_vadd_left (v : V) (x : P) : dist (v +ᵥ x) x = ‖v‖ := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v : V
    x : P
    ⊢ Eq (Dist.dist (HVAdd.hVAdd v x) x) (Norm.norm v)
  -/
  simp [dist_eq_norm_vsub V _ x]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_vadd_left (v : V) (x : P) : nndist (v +ᵥ x) x = ‖v‖₊ :=
  NNReal.eq <| dist_vadd_left _ _


@[simp]
                                                                      /-
                                                                        V : Type u_2
                                                                        P : Type u_3
                                                                        inst✝² : SeminormedAddCommGroup V
                                                                        inst✝¹ : PseudoMetricSpace P
                                                                        inst✝ : NormedAddTorsor V P
                                                                        v : V
                                                                        x : P
                                                                        ⊢ Eq (Dist.dist x (HVAdd.hVAdd v x)) (Norm.norm v)
                                                                      -/
theorem dist_vadd_right (v : V) (x : P) : dist x (v +ᵥ x) = ‖v‖ := by rw [dist_comm, dist_vadd_left]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem nndist_vadd_right (v : V) (x : P) : nndist x (v +ᵥ x) = ‖v‖₊ :=
  NNReal.eq <| dist_vadd_right _ _


/-- Isometry between the tangent space `V` of a (semi)normed add torsor `P` and `P` given by
addition/subtraction of `x : P`. -/
@[simps!]
def IsometryEquiv.vaddConst (x : P) : V ≃ᵢ P where
  toEquiv := Equiv.vaddConst x
  isometry_toFun := Isometry.of_dist_eq fun _ _ => dist_vadd_cancel_right _ _ _


@[simp]
theorem dist_vsub_cancel_left (x y z : P) : dist (x -ᵥ y) (x -ᵥ z) = dist y z := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    x y z : P
    ⊢ Eq (Dist.dist (VSub.vsub x y) (VSub.vsub x z)) (Dist.dist y z)
  -/
  rw [dist_eq_norm, vsub_sub_vsub_cancel_left, dist_comm, dist_eq_norm_vsub V]
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_vsub_cancel_left (x y z : P) : nndist (x -ᵥ y) (x -ᵥ z) = nndist y z :=
  NNReal.eq <| dist_vsub_cancel_left _ _ _


/-- Isometry between the tangent space `V` of a (semi)normed add torsor `P` and `P` given by
subtraction from `x : P`. -/
@[simps!]
def IsometryEquiv.constVSub (x : P) : P ≃ᵢ V where
  toEquiv := Equiv.constVSub x
  isometry_toFun := Isometry.of_dist_eq fun _ _ => dist_vsub_cancel_left _ _ _


@[simp]
theorem dist_vsub_cancel_right (x y z : P) : dist (x -ᵥ z) (y -ᵥ z) = dist x y :=
  (IsometryEquiv.vaddConst z).symm.dist_eq x y


@[simp]
theorem nndist_vsub_cancel_right (x y z : P) : nndist (x -ᵥ z) (y -ᵥ z) = nndist x y :=
  NNReal.eq <| dist_vsub_cancel_right _ _ _


theorem dist_vadd_vadd_le (v v' : V) (p p' : P) :
    dist (v +ᵥ p) (v' +ᵥ p') ≤ dist v v' + dist p p' := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v v' : V
    p p' : P
    ⊢ LE.le (Dist.dist (HVAdd.hVAdd v p) (HVAdd.hVAdd v' p')) (HAdd.hAdd (Dist.dis …
  -/
  simpa using dist_triangle (v +ᵥ p) (v' +ᵥ p) (v' +ᵥ p')
  /-
    🎉 no goals
  -/


theorem nndist_vadd_vadd_le (v v' : V) (p p' : P) :
    nndist (v +ᵥ p) (v' +ᵥ p') ≤ nndist v v' + nndist p p' :=
  dist_vadd_vadd_le _ _ _ _


theorem dist_vsub_vsub_le (p₁ p₂ p₃ p₄ : P) :
    dist (p₁ -ᵥ p₂) (p₃ -ᵥ p₄) ≤ dist p₁ p₃ + dist p₂ p₄ := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (Dist.dist (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄)) (HAdd.hAdd (Dist.dist  …
  -/
  rw [dist_eq_norm, vsub_sub_vsub_comm, dist_eq_norm_vsub V, dist_eq_norm_vsub V]
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (Norm.norm (HSub.hSub (VSub.vsub p₁ p₃) (VSub.vsub p₂ p₄))) (HAdd.hAdd …
  -/
  exact norm_sub_le _ _
  /-
    🎉 no goals
  -/


theorem nndist_vsub_vsub_le (p₁ p₂ p₃ p₄ : P) :
    nndist (p₁ -ᵥ p₂) (p₃ -ᵥ p₄) ≤ nndist p₁ p₃ + nndist p₂ p₄ := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (NNDist.nndist (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄)) (HAdd.hAdd (NNDist …
  -/
  simp only [← NNReal.coe_le_coe, NNReal.coe_add, ← dist_nndist, dist_vsub_vsub_le]
  /-
    🎉 no goals
  -/


theorem edist_vadd_vadd_le (v v' : V) (p p' : P) :
    edist (v +ᵥ p) (v' +ᵥ p') ≤ edist v v' + edist p p' := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v v' : V
    p p' : P
    ⊢ LE.le (EDist.edist (HVAdd.hVAdd v p) (HVAdd.hVAdd v' p')) (HAdd.hAdd (EDist. …
  -/
  simp only [edist_nndist]
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v v' : V
    p p' : P
    ⊢ LE.le (↑(NNDist.nndist (HVAdd.hVAdd v p) (HVAdd.hVAdd v' p'))) (HAdd.hAdd ↑( …
  -/
  norm_cast  -- Porting note: was apply_mod_cast
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    v v' : V
    p p' : P
    ⊢ LE.le (NNDist.nndist (HVAdd.hVAdd v p) (HVAdd.hVAdd v' p')) (HAdd.hAdd (NNDi …
  -/
  apply dist_vadd_vadd_le
  /-
    🎉 no goals
  -/


theorem edist_vsub_vsub_le (p₁ p₂ p₃ p₄ : P) :
    edist (p₁ -ᵥ p₂) (p₃ -ᵥ p₄) ≤ edist p₁ p₃ + edist p₂ p₄ := by
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (EDist.edist (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄)) (HAdd.hAdd (EDist.ed …
  -/
  simp only [edist_nndist]
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (↑(NNDist.nndist (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄))) (HAdd.hAdd ↑(NN …
  -/
  norm_cast  -- Porting note: was apply_mod_cast
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ p₄ : P
    ⊢ LE.le (NNDist.nndist (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄)) (HAdd.hAdd (NNDist …
  -/
  apply dist_vsub_vsub_le
  /-
    🎉 no goals
  -/


/-- The pseudodistance defines a pseudometric space structure on the torsor. This
is not an instance because it depends on `V` to define a `MetricSpace P`. -/
def pseudoMetricSpaceOfNormedAddCommGroupOfAddTorsor (V P : Type*) [SeminormedAddCommGroup V]
    [AddTorsor V P] : PseudoMetricSpace P where
  dist x y := ‖(x -ᵥ y : V)‖
                    /-
                      α : Type u_1
                      V✝ : Type u_2
                      P✝ : Type u_3
                      W : Type u_4
                      Q : Type u_5
                      inst✝⁷ : SeminormedAddCommGroup V✝
                      inst✝⁶ : PseudoMetricSpace P✝
                      inst✝⁵ : NormedAddTorsor V✝ P✝
                      inst✝⁴ : NormedAddCommGroup W
                      inst✝³ : MetricSpace Q
                      inst✝² : NormedAddTorsor W Q
                      V : Type u_6
                      P : Type u_7
                      inst✝¹ : SeminormedAddCommGroup V
                      inst✝ : AddTorsor V P
                      x : P
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp
                    /-
                      🎉 no goals
                    -/
                      /-
                        α : Type u_1
                        V✝ : Type u_2
                        P✝ : Type u_3
                        W : Type u_4
                        Q : Type u_5
                        inst✝⁷ : SeminormedAddCommGroup V✝
                        inst✝⁶ : PseudoMetricSpace P✝
                        inst✝⁵ : NormedAddTorsor V✝ P✝
                        inst✝⁴ : NormedAddCommGroup W
                        inst✝³ : MetricSpace Q
                        inst✝² : NormedAddTorsor W Q
                        V : Type u_6
                        P : Type u_7
                        inst✝¹ : SeminormedAddCommGroup V
                        inst✝ : AddTorsor V P
                        x y : P
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  dist_comm x y := by simp only [← neg_vsub_eq_vsub_rev y x, norm_neg]
                      /-
                        🎉 no goals
                      -/
  dist_triangle x y z := by
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    change ‖x -ᵥ z‖ ≤ ‖x -ᵥ y‖ + ‖y -ᵥ z‖
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Norm.norm (VSub.vsub x z)) (HAdd.hAdd (Norm.norm (VSub.vsub x y)) (No …
    -/
    rw [← vsub_add_vsub_cancel]
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Norm.norm (HAdd.hAdd (VSub.vsub x ?p₂) (VSub.vsub ?p₂ z))) (HAdd.hAdd …
    -/
    apply norm_add_le
    /-
      🎉 no goals
    -/


/-- The distance defines a metric space structure on the torsor. This
is not an instance because it depends on `V` to define a `MetricSpace P`. -/
def metricSpaceOfNormedAddCommGroupOfAddTorsor (V P : Type*) [NormedAddCommGroup V]
    [AddTorsor V P] : MetricSpace P where
  dist x y := ‖(x -ᵥ y : V)‖
                    /-
                      α : Type u_1
                      V✝ : Type u_2
                      P✝ : Type u_3
                      W : Type u_4
                      Q : Type u_5
                      inst✝⁷ : SeminormedAddCommGroup V✝
                      inst✝⁶ : PseudoMetricSpace P✝
                      inst✝⁵ : NormedAddTorsor V✝ P✝
                      inst✝⁴ : NormedAddCommGroup W
                      inst✝³ : MetricSpace Q
                      inst✝² : NormedAddTorsor W Q
                      V : Type u_6
                      P : Type u_7
                      inst✝¹ : NormedAddCommGroup V
                      inst✝ : AddTorsor V P
                      x : P
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp
                    /-
                      🎉 no goals
                    -/
                             /-
                               α : Type u_1
                               V✝ : Type u_2
                               P✝ : Type u_3
                               W : Type u_4
                               Q : Type u_5
                               inst✝⁷ : SeminormedAddCommGroup V✝
                               inst✝⁶ : PseudoMetricSpace P✝
                               inst✝⁵ : NormedAddTorsor V✝ P✝
                               inst✝⁴ : NormedAddCommGroup W
                               inst✝³ : MetricSpace Q
                               inst✝² : NormedAddTorsor W Q
                               V : Type u_6
                               P : Type u_7
                               inst✝¹ : NormedAddCommGroup V
                               inst✝ : AddTorsor V P
                               x✝ y✝ : P
                               h : Eq (Dist.dist x✝ y✝) 0
                               ⊢ Eq x✝ y✝
                             -/
                      /-
                        α : Type u_1
                        V✝ : Type u_2
                        P✝ : Type u_3
                        W : Type u_4
                        Q : Type u_5
                        inst✝⁷ : SeminormedAddCommGroup V✝
                        inst✝⁶ : PseudoMetricSpace P✝
                        inst✝⁵ : NormedAddTorsor V✝ P✝
                        inst✝⁴ : NormedAddCommGroup W
                        inst✝³ : MetricSpace Q
                        inst✝² : NormedAddTorsor W Q
                        V : Type u_6
                        P : Type u_7
                        inst✝¹ : NormedAddCommGroup V
                        inst✝ : AddTorsor V P
                        x y : P
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  eq_of_dist_eq_zero h := by simpa using h
                      /-
                        🎉 no goals
                      -/
                             /-
                               🎉 no goals
                             -/
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : NormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
  dist_comm x y := by simp only [← neg_vsub_eq_vsub_rev y x, norm_neg]
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : NormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Norm.norm (VSub.vsub x z)) (HAdd.hAdd (Norm.norm (VSub.vsub x y)) (No …
    -/
  dist_triangle x y z := by
    /-
      α : Type u_1
      V✝ : Type u_2
      P✝ : Type u_3
      W : Type u_4
      Q : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V✝
      inst✝⁶ : PseudoMetricSpace P✝
      inst✝⁵ : NormedAddTorsor V✝ P✝
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : MetricSpace Q
      inst✝² : NormedAddTorsor W Q
      V : Type u_6
      P : Type u_7
      inst✝¹ : NormedAddCommGroup V
      inst✝ : AddTorsor V P
      x y z : P
      ⊢ LE.le (Norm.norm (HAdd.hAdd (VSub.vsub x ?p₂) (VSub.vsub ?p₂ z))) (HAdd.hAdd …
    -/
    change ‖x -ᵥ z‖ ≤ ‖x -ᵥ y‖ + ‖y -ᵥ z‖
    /-
      🎉 no goals
    -/
    rw [← vsub_add_vsub_cancel]
    apply norm_add_le


theorem LipschitzWith.vadd [PseudoEMetricSpace α] {f : α → V} {g : α → P} {Kf Kg : ℝ≥0}
    (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) : LipschitzWith (Kf + Kg) (f +ᵥ g) :=
  fun x y =>
  calc
    edist (f x +ᵥ g x) (f y +ᵥ g y) ≤ edist (f x) (f y) + edist (g x) (g y) :=
      edist_vadd_vadd_le _ _ _ _
    _ ≤ Kf * edist x y + Kg * edist x y := add_le_add (hf x y) (hg x y)
    _ = (Kf + Kg) * edist x y := (add_mul _ _ _).symm


theorem LipschitzWith.vsub [PseudoEMetricSpace α] {f g : α → P} {Kf Kg : ℝ≥0}
    (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) : LipschitzWith (Kf + Kg) (f -ᵥ g) :=
  fun x y =>
  calc
    edist (f x -ᵥ g x) (f y -ᵥ g y) ≤ edist (f x) (f y) + edist (g x) (g y) :=
      edist_vsub_vsub_le _ _ _ _
    _ ≤ Kf * edist x y + Kg * edist x y := add_le_add (hf x y) (hg x y)
    _ = (Kf + Kg) * edist x y := (add_mul _ _ _).symm


theorem uniformContinuous_vadd : UniformContinuous fun x : V × P => x.1 +ᵥ x.2 :=
  (LipschitzWith.prod_fst.vadd LipschitzWith.prod_snd).uniformContinuous


theorem uniformContinuous_vsub : UniformContinuous fun x : P × P => x.1 -ᵥ x.2 :=
  (LipschitzWith.prod_fst.vsub LipschitzWith.prod_snd).uniformContinuous


instance (priority := 100) NormedAddTorsor.to_continuousVAdd : ContinuousVAdd V P where
  continuous_vadd := uniformContinuous_vadd.continuous


theorem continuous_vsub : Continuous fun x : P × P => x.1 -ᵥ x.2 :=
  uniformContinuous_vsub.continuous


theorem Filter.Tendsto.vsub {l : Filter α} {f g : α → P} {x y : P} (hf : Tendsto f l (𝓝 x))
    (hg : Tendsto g l (𝓝 y)) : Tendsto (f -ᵥ g) l (𝓝 (x -ᵥ y)) :=
  (continuous_vsub.tendsto (x, y)).comp (hf.prod_mk_nhds hg)


theorem Continuous.vsub {f g : α → P} (hf : Continuous f) (hg : Continuous g) :
    Continuous (f -ᵥ g) :=
  continuous_vsub.comp (hf.prod_mk hg : _)


nonrec theorem ContinuousAt.vsub {f g : α → P} {x : α} (hf : ContinuousAt f x)
    (hg : ContinuousAt g x) :
    ContinuousAt (f -ᵥ g) x :=
  hf.vsub hg


nonrec theorem ContinuousWithinAt.vsub {f g : α → P} {x : α} {s : Set α}
    (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (f -ᵥ g) s x :=
  hf.vsub hg


theorem ContinuousOn.vsub {f g : α → P} {s : Set α} (hf : ContinuousOn f s)
    (hg : ContinuousOn g s) : ContinuousOn (f -ᵥ g) s := fun x hx ↦
  (hf x hx).vsub (hg x hx)


theorem Filter.Tendsto.lineMap {l : Filter α} {f₁ f₂ : α → P} {g : α → R} {p₁ p₂ : P} {c : R}
    (h₁ : Tendsto f₁ l (𝓝 p₁)) (h₂ : Tendsto f₂ l (𝓝 p₂)) (hg : Tendsto g l (𝓝 c)) :
    Tendsto (fun x => AffineMap.lineMap (f₁ x) (f₂ x) (g x)) l (𝓝 <| AffineMap.lineMap p₁ p₂ c) :=
  (hg.smul (h₂.vsub h₁)).vadd h₁


theorem Filter.Tendsto.midpoint [Invertible (2 : R)] {l : Filter α} {f₁ f₂ : α → P} {p₁ p₂ : P}
    (h₁ : Tendsto f₁ l (𝓝 p₁)) (h₂ : Tendsto f₂ l (𝓝 p₂)) :
    Tendsto (fun x => midpoint R (f₁ x) (f₂ x)) l (𝓝 <| midpoint R p₁ p₂) :=
  h₁.lineMap h₂ tendsto_const_nhds


theorem IsClosed.vadd_right_of_isCompact {s : Set V} {t : Set P} (hs : IsClosed s)
    (ht : IsCompact t) : IsClosed (s +ᵥ t) := by
  -- This result is still true for any `AddTorsor` where `-ᵥ` is continuous,
  -- but we don't yet have a nice way to state it.
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set V
    t : Set P
    hs : IsClosed s
    ht : IsCompact t
    ⊢ IsClosed (HVAdd.hVAdd s t)
  -/
  refine IsSeqClosed.isClosed (fun u p husv hup ↦ ?_)
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set V
    t : Set P
    hs : IsClosed s
    ht : IsCompact t
    u : Nat → P
    p : P
    husv : ∀ (n : Nat), Membership.mem (HVAdd.hVAdd s t) (u n)
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    ⊢ Membership.mem (HVAdd.hVAdd s t) p
  -/
  choose! a ha v hv hav using husv
  /-
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set V
    t : Set P
    hs : IsClosed s
    ht : IsCompact t
    u : Nat → P
    p : P
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    a : Nat → V
    ha : ∀ (n : Nat), Membership.mem s (a n)
    v : Nat → P
    hv : ∀ (n : Nat), Membership.mem t (v n)
    hav : ∀ (n : Nat), Eq ((fun x1 x2 => HVAdd.hVAdd x1 x2) (a n) (v n)) (u n)
    ⊢ Membership.mem (HVAdd.hVAdd s t) p
  -/
  rcases ht.isSeqCompact hv with ⟨q, hqt, φ, φ_mono, hφq⟩
  refine ⟨p -ᵥ q, hs.mem_of_tendsto ((hup.comp φ_mono.tendsto_atTop).vsub hφq)
    (Eventually.of_forall fun n ↦ ?_), q, hqt, vsub_vadd _ _⟩
  /-
    case intro.intro.intro.intro
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set V
    t : Set P
    hs : IsClosed s
    ht : IsCompact t
    u : Nat → P
    p : P
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    a : Nat → V
    ha : ∀ (n : Nat), Membership.mem s (a n)
    v : Nat → P
    hv : ∀ (n : Nat), Membership.mem t (v n)
    hav : ∀ (n : Nat), Eq ((fun x1 x2 => HVAdd.hVAdd x1 x2) (a n) (v n)) (u n)
    q : P
    hqt : Membership.mem t q
    φ : Nat → Nat
    φ_mono : StrictMono φ
    hφq : Filter.Tendsto (Function.comp v φ) Filter.atTop (nhds q)
    n : Nat
    ⊢ Membership.mem s (VSub.vsub (Function.comp u φ) (Function.comp v φ) n)
  -/
  convert ha (φ n) using 1
  /-
    case h.e'_5
    V : Type u_2
    P : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set V
    t : Set P
    hs : IsClosed s
    ht : IsCompact t
    u : Nat → P
    p : P
    hup : Filter.Tendsto u Filter.atTop (nhds p)
    a : Nat → V
    ha : ∀ (n : Nat), Membership.mem s (a n)
    v : Nat → P
    hv : ∀ (n : Nat), Membership.mem t (v n)
    hav : ∀ (n : Nat), Eq ((fun x1 x2 => HVAdd.hVAdd x1 x2) (a n) (v n)) (u n)
    q : P
    hqt : Membership.mem t q
    φ : Nat → Nat
    φ_mono : StrictMono φ
    hφq : Filter.Tendsto (Function.comp v φ) Filter.atTop (nhds q)
    n : Nat
    ⊢ Eq (VSub.vsub (Function.comp u φ) (Function.comp v φ) n) (a (φ n))
  -/
  exact (eq_vadd_iff_vsub_eq _ _ _).mp (hav (φ n)).symm
  /-
    🎉 no goals
  -/


