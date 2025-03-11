/-- An additive action such that for all `c`, the map `fun x ↦ c +ᵥ x` is uniformly continuous. -/
class UniformContinuousConstVAdd [VAdd M X] : Prop where
  uniformContinuous_const_vadd : ∀ c : M, UniformContinuous (c +ᵥ · : X → X)


/-- A multiplicative action such that for all `c`,
the map `fun x ↦ c • x` is uniformly continuous. -/
@[to_additive]
class UniformContinuousConstSMul [SMul M X] : Prop where
  uniformContinuous_const_smul : ∀ c : M, UniformContinuous (c • · : X → X)


instance AddMonoid.uniformContinuousConstSMul_nat [AddGroup X] [UniformAddGroup X] :
    UniformContinuousConstSMul ℕ X :=
  ⟨uniformContinuous_const_nsmul⟩


instance AddGroup.uniformContinuousConstSMul_int [AddGroup X] [UniformAddGroup X] :
    UniformContinuousConstSMul ℤ X :=
  ⟨uniformContinuous_const_zsmul⟩


/-- A `DistribMulAction` that is continuous on a uniform group is uniformly continuous.
This can't be an instance due to it forming a loop with
`UniformContinuousConstSMul.to_continuousConstSMul` -/
theorem uniformContinuousConstSMul_of_continuousConstSMul [Monoid R] [AddCommGroup M]
    [DistribMulAction R M] [UniformSpace M] [UniformAddGroup M] [ContinuousConstSMul R M] :
    UniformContinuousConstSMul R M :=
  ⟨fun r =>
    uniformContinuous_of_continuousAt_zero (DistribMulAction.toAddMonoidHom M r)
      (Continuous.continuousAt (continuous_const_smul r))⟩


/-- The action of `Semiring.toModule` is uniformly continuous. -/
instance Ring.uniformContinuousConstSMul [Ring R] [UniformSpace R] [UniformAddGroup R]
    [ContinuousMul R] : UniformContinuousConstSMul R R :=
  uniformContinuousConstSMul_of_continuousConstSMul _ _


/-- The action of `Semiring.toOppositeModule` is uniformly continuous. -/
instance Ring.uniformContinuousConstSMul_op [Ring R] [UniformSpace R] [UniformAddGroup R]
    [ContinuousMul R] : UniformContinuousConstSMul Rᵐᵒᵖ R :=
  uniformContinuousConstSMul_of_continuousConstSMul _ _


@[to_additive]
instance (priority := 100) UniformContinuousConstSMul.to_continuousConstSMul
    [UniformContinuousConstSMul M X] : ContinuousConstSMul M X :=
  ⟨fun c => (uniformContinuous_const_smul c).continuous⟩


@[to_additive]
theorem UniformContinuous.const_smul [UniformContinuousConstSMul M X] {f : Y → X}
    (hf : UniformContinuous f) (c : M) : UniformContinuous (c • f) :=
  (uniformContinuous_const_smul c).comp hf


@[to_additive]
lemma IsUniformInducing.uniformContinuousConstSMul [SMul M Y] [UniformContinuousConstSMul M Y]
    {f : X → Y} (hf : IsUniformInducing f) (hsmul : ∀ (c : M) x, f (c • x) = c • f x) :
    UniformContinuousConstSMul M X where
  uniformContinuous_const_smul c := by
    simpa only [hf.uniformContinuous_iff, Function.comp_def, hsmul]
      using hf.uniformContinuous.const_smul c


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformContinuousConstSMul := IsUniformInducing.uniformContinuousConstSMul


/-- If a scalar action is central, then its right action is uniform continuous when its left action
is. -/
@[to_additive "If an additive action is central, then its right action is uniform
continuous when its left action is."]
instance (priority := 100) UniformContinuousConstSMul.op [SMul Mᵐᵒᵖ X] [IsCentralScalar M X]
    [UniformContinuousConstSMul M X] : UniformContinuousConstSMul Mᵐᵒᵖ X :=
                               /-
                                 R : Type u
                                 M : Type v
                                 N : Type w
                                 X : Type x
                                 Y : Type y
                                 inst✝⁵ : UniformSpace X
                                 inst✝⁴ : UniformSpace Y
                                 inst✝³ : SMul M X
                                 inst✝² : SMul (MulOpposite M) X
                                 inst✝¹ : IsCentralScalar M X
                                 inst✝ : UniformContinuousConstSMul M X
                                 c : M
                                 ⊢ UniformContinuous fun x => HSMul.hSMul (MulOpposite.op c) x
                               -/
  ⟨MulOpposite.rec' fun c ↦ by simpa only [op_smul_eq_smul] using uniformContinuous_const_smul c⟩
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
instance MulOpposite.uniformContinuousConstSMul [UniformContinuousConstSMul M X] :
    UniformContinuousConstSMul M Xᵐᵒᵖ :=
  ⟨fun c =>
    MulOpposite.uniformContinuous_op.comp <| MulOpposite.uniformContinuous_unop.const_smul c⟩


@[to_additive]
instance UniformGroup.to_uniformContinuousConstSMul {G : Type u} [Group G] [UniformSpace G]
    [UniformGroup G] : UniformContinuousConstSMul G G :=
  ⟨fun _ => uniformContinuous_const.mul uniformContinuous_id⟩


theorem UniformContinuous.const_mul' [UniformContinuousConstSMul R R] {f : β → R}
    (hf : UniformContinuous f) (a : R) : UniformContinuous fun x ↦ a * f x :=
  hf.const_smul a


theorem UniformContinuous.mul_const' [UniformContinuousConstSMul Rᵐᵒᵖ R] {f : β → R}
    (hf : UniformContinuous f) (a : R) : UniformContinuous fun x ↦ f x * a :=
  hf.const_smul (MulOpposite.op a)


theorem uniformContinuous_mul_left' [UniformContinuousConstSMul R R] (a : R) :
    UniformContinuous fun b : R => a * b :=
  uniformContinuous_id.const_mul' _


theorem uniformContinuous_mul_right' [UniformContinuousConstSMul Rᵐᵒᵖ R] (a : R) :
    UniformContinuous fun b : R => b * a :=
  uniformContinuous_id.mul_const' _


theorem UniformContinuous.div_const' {R β : Type*} [DivisionRing R] [UniformSpace R]
    [UniformContinuousConstSMul Rᵐᵒᵖ R] [UniformSpace β] {f : β → R}
    (hf : UniformContinuous f) (a : R) :
    UniformContinuous fun x ↦ f x / a := by
  /-
    R : Type u_3
    β : Type u_4
    inst✝³ : DivisionRing R
    inst✝² : UniformSpace R
    inst✝¹ : UniformContinuousConstSMul (MulOpposite R) R
    inst✝ : UniformSpace β
    f : β → R
    hf : UniformContinuous f
    a : R
    ⊢ UniformContinuous fun x => HDiv.hDiv (f x) a
  -/
  simpa [div_eq_mul_inv] using hf.mul_const' a⁻¹
  /-
    🎉 no goals
  -/


theorem uniformContinuous_div_const' {R : Type*} [DivisionRing R] [UniformSpace R]
    [UniformContinuousConstSMul Rᵐᵒᵖ R] (a : R) :
    UniformContinuous fun b : R => b / a :=
  uniformContinuous_id.div_const' _


@[to_additive]
noncomputable instance : SMul M (Completion X) :=
  ⟨fun c => Completion.map (c • ·)⟩


@[to_additive]
theorem smul_def (c : M) (x : Completion X) : c • x = Completion.map (c • ·) x :=
  rfl


@[to_additive]
instance : UniformContinuousConstSMul M (Completion X) :=
  ⟨fun _ => uniformContinuous_map⟩


@[to_additive instVAddAssocClass]
instance instIsScalarTower [SMul N X] [SMul M N] [UniformContinuousConstSMul M X]
    [UniformContinuousConstSMul N X] [IsScalarTower M N X] : IsScalarTower M N (Completion X) :=
  ⟨fun m n x => by
    have : _ = (_ : Completion X → Completion X) :=
      map_comp (uniformContinuous_const_smul m) (uniformContinuous_const_smul n)
    /-
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁷ : UniformSpace X
      inst✝⁶ : UniformSpace Y
      inst✝⁵ : SMul M X
      inst✝⁴ : SMul N X
      inst✝³ : SMul M N
      inst✝² : UniformContinuousConstSMul M X
      inst✝¹ : UniformContinuousConstSMul N X
      inst✝ : IsScalarTower M N X
      m : M
      n : N
      x : UniformSpace.Completion X
      this : Eq (Function.comp (UniformSpace.Completion.map fun x => HSMul.hSMul m x …
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul m n) x) (HSMul.hSMul m (HSMul.hSMul n x))
    -/
    refine Eq.trans ?_ (congr_fun this.symm x)
    /-
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁷ : UniformSpace X
      inst✝⁶ : UniformSpace Y
      inst✝⁵ : SMul M X
      inst✝⁴ : SMul N X
      inst✝³ : SMul M N
      inst✝² : UniformContinuousConstSMul M X
      inst✝¹ : UniformContinuousConstSMul N X
      inst✝ : IsScalarTower M N X
      m : M
      n : N
      x : UniformSpace.Completion X
      this : Eq (Function.comp (UniformSpace.Completion.map fun x => HSMul.hSMul m x …
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul m n) x) (UniformSpace.Completion.map (Function. …
    -/
    exact congr_arg (fun f => Completion.map f x) (funext (smul_assoc _ _))⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMul N X] [SMulCommClass M N X] [UniformContinuousConstSMul M X]
    [UniformContinuousConstSMul N X] : SMulCommClass M N (Completion X) :=
  ⟨fun m n x => by
    /-
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁶ : UniformSpace X
      inst✝⁵ : UniformSpace Y
      inst✝⁴ : SMul M X
      inst✝³ : SMul N X
      inst✝² : SMulCommClass M N X
      inst✝¹ : UniformContinuousConstSMul M X
      inst✝ : UniformContinuousConstSMul N X
      m : M
      n : N
      x : UniformSpace.Completion X
      ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n x)) (HSMul.hSMul n (HSMul.hSMul m x))
    -/
    have hmn : m • n • x = (Completion.map (SMul.smul m) ∘ Completion.map (SMul.smul n)) x := rfl
    /-
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁶ : UniformSpace X
      inst✝⁵ : UniformSpace Y
      inst✝⁴ : SMul M X
      inst✝³ : SMul N X
      inst✝² : SMulCommClass M N X
      inst✝¹ : UniformContinuousConstSMul M X
      inst✝ : UniformContinuousConstSMul N X
      m : M
      n : N
      x : UniformSpace.Completion X
      hmn : Eq (HSMul.hSMul m (HSMul.hSMul n x)) (Function.comp (UniformSpace.Comple …
      ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n x)) (HSMul.hSMul n (HSMul.hSMul m x))
    -/
    have hnm : n • m • x = (Completion.map (SMul.smul n) ∘ Completion.map (SMul.smul m)) x := rfl
    /-
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁶ : UniformSpace X
      inst✝⁵ : UniformSpace Y
      inst✝⁴ : SMul M X
      inst✝³ : SMul N X
      inst✝² : SMulCommClass M N X
      inst✝¹ : UniformContinuousConstSMul M X
      inst✝ : UniformContinuousConstSMul N X
      m : M
      n : N
      x : UniformSpace.Completion X
      hmn : Eq (HSMul.hSMul m (HSMul.hSMul n x)) (Function.comp (UniformSpace.Comple …
      hnm : Eq (HSMul.hSMul n (HSMul.hSMul m x)) (Function.comp (UniformSpace.Comple …
      ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n x)) (HSMul.hSMul n (HSMul.hSMul m x))
    -/
    rw [hmn, hnm, map_comp, map_comp]
      /-
        R : Type u
        M : Type v
        N : Type w
        X : Type x
        Y : Type y
        inst✝⁶ : UniformSpace X
        inst✝⁵ : UniformSpace Y
        inst✝⁴ : SMul M X
        inst✝³ : SMul N X
        inst✝² : SMulCommClass M N X
        inst✝¹ : UniformContinuousConstSMul M X
        inst✝ : UniformContinuousConstSMul N X
        m : M
        n : N
        x : UniformSpace.Completion X
        hmn : Eq (HSMul.hSMul m (HSMul.hSMul n x)) (Function.comp (UniformSpace.Comple …
        hnm : Eq (HSMul.hSMul n (HSMul.hSMul m x)) (Function.comp (UniformSpace.Comple …
        ⊢ Eq (UniformSpace.Completion.map (Function.comp (SMul.smul m) (SMul.smul n))  …
      -/
    · exact congr_arg (fun f => Completion.map f x) (funext (smul_comm _ _))
      /-
        🎉 no goals
      -/
    /-
      case hg
      R : Type u
      M : Type v
      N : Type w
      X : Type x
      Y : Type y
      inst✝⁶ : UniformSpace X
      inst✝⁵ : UniformSpace Y
      inst✝⁴ : SMul M X
      inst✝³ : SMul N X
      inst✝² : SMulCommClass M N X
      inst✝¹ : UniformContinuousConstSMul M X
      inst✝ : UniformContinuousConstSMul N X
      m : M
      n : N
      x : UniformSpace.Completion X
      hmn : Eq (HSMul.hSMul m (HSMul.hSMul n x)) (Function.comp (UniformSpace.Comple …
      hnm : Eq (HSMul.hSMul n (HSMul.hSMul m x)) (Function.comp (UniformSpace.Comple …
      ⊢ UniformContinuous (SMul.smul n)
    -/
    repeat' exact uniformContinuous_const_smul _⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMul Mᵐᵒᵖ X] [IsCentralScalar M X] : IsCentralScalar M (Completion X) :=
  ⟨fun c a => (congr_arg fun f => Completion.map f a) <| funext (op_smul_eq_smul c)⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_smul (c : M) (x : X) : (↑(c • x) : Completion X) = c • (x : Completion X) :=
  (map_coe (uniformContinuous_const_smul c) x).symm


@[to_additive]
noncomputable instance [Monoid M] [MulAction M X] [UniformContinuousConstSMul M X] :
    MulAction M (Completion X) where
  smul := (· • ·)
                                                                       /-
                                                                         R : Type u
                                                                         M : Type v
                                                                         N : Type w
                                                                         X : Type x
                                                                         Y : Type y
                                                                         inst✝⁴ : UniformSpace X
                                                                         inst✝³ : UniformSpace Y
                                                                         inst✝² : Monoid M
                                                                         inst✝¹ : MulAction M X
                                                                         inst✝ : UniformContinuousConstSMul M X
                                                                         a : X
                                                                         ⊢ Eq (HSMul.hSMul 1 (↑X a)) (↑X a)
                                                                       -/
  one_smul := ext' (continuous_const_smul _) continuous_id fun a => by rw [← coe_smul, one_smul]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  mul_smul x y :=
    ext' (continuous_const_smul _) ((continuous_const_smul _).const_smul _) fun a => by
      /-
        R : Type u
        M : Type v
        N : Type w
        X : Type x
        Y : Type y
        inst✝⁴ : UniformSpace X
        inst✝³ : UniformSpace Y
        inst✝² : Monoid M
        inst✝¹ : MulAction M X
        inst✝ : UniformContinuousConstSMul M X
        x y : M
        a : X
        ⊢ Eq (HSMul.hSMul (HMul.hMul x y) (↑X a)) (HSMul.hSMul x (HSMul.hSMul y (↑X a)))
      -/
      simp only [← coe_smul, mul_smul]
      /-
        🎉 no goals
      -/


