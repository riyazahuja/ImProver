@[simp]
theorem tmul_comp_tensorMap
    {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂}
    {Q₃ : QuadraticForm R M₃} {Q₄ : QuadraticForm R M₄}
    (f : Q₁ →qᵢ Q₂) (g : Q₃ →qᵢ Q₄) :
    (Q₂.tmul Q₄).comp (TensorProduct.map f.toLinearMap g.toLinearMap) = Q₁.tmul Q₃ := by
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    M₄ : Type uM₄
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : AddCommGroup M₄
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Module R M₃
    inst✝¹ : Module R M₄
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    Q₄ : QuadraticForm R M₄
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₃ Q₄
    ⊢ Eq (QuadraticMap.comp (Q₂.tmul Q₄) (TensorProduct.map f.toLinearMap g.toLine …
  -/
  have h₁ : Q₁ = Q₂.comp f.toLinearMap := QuadraticMap.ext fun x => (f.map_app x).symm
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    M₄ : Type uM₄
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : AddCommGroup M₄
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Module R M₃
    inst✝¹ : Module R M₄
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    Q₄ : QuadraticForm R M₄
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₃ Q₄
    h₁ : Eq Q₁ (QuadraticMap.comp Q₂ f.toLinearMap)
    ⊢ Eq (QuadraticMap.comp (Q₂.tmul Q₄) (TensorProduct.map f.toLinearMap g.toLine …
  -/
  have h₃ : Q₃ = Q₄.comp g.toLinearMap := QuadraticMap.ext fun x => (g.map_app x).symm
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    M₄ : Type uM₄
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : AddCommGroup M₄
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Module R M₃
    inst✝¹ : Module R M₄
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    Q₄ : QuadraticForm R M₄
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₃ Q₄
    h₁ : Eq Q₁ (QuadraticMap.comp Q₂ f.toLinearMap)
    h₃ : Eq Q₃ (QuadraticMap.comp Q₄ g.toLinearMap)
    ⊢ Eq (QuadraticMap.comp (Q₂.tmul Q₄) (TensorProduct.map f.toLinearMap g.toLine …
  -/
  refine (QuadraticMap.associated_rightInverse R).injective ?_
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    M₄ : Type uM₄
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : AddCommGroup M₄
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Module R M₃
    inst✝¹ : Module R M₄
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    Q₄ : QuadraticForm R M₄
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₃ Q₄
    h₁ : Eq Q₁ (QuadraticMap.comp Q₂ f.toLinearMap)
    h₃ : Eq Q₃ (QuadraticMap.comp Q₄ g.toLinearMap)
    ⊢ Eq ((QuadraticMap.associatedHom R) (QuadraticMap.comp (Q₂.tmul Q₄) (TensorPr …
  -/
  ext m₁ m₃ m₁' m₃'
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    M₄ : Type uM₄
    inst✝⁹ : CommRing R
    inst✝⁸ : AddCommGroup M₁
    inst✝⁷ : AddCommGroup M₂
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : AddCommGroup M₄
    inst✝⁴ : Module R M₁
    inst✝³ : Module R M₂
    inst✝² : Module R M₃
    inst✝¹ : Module R M₄
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    Q₄ : QuadraticForm R M₄
    f : QuadraticMap.Isometry Q₁ Q₂
    g : QuadraticMap.Isometry Q₃ Q₄
    h₁ : Eq Q₁ (QuadraticMap.comp Q₂ f.toLinearMap)
    h₃ : Eq Q₃ (QuadraticMap.comp Q₄ g.toLinearMap)
    m₁ : M₁
    m₃ : M₃
    m₁' : M₁
    m₃' : M₃
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  simp [-associated_apply, h₁, h₃, associated_tmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem tmul_tensorMap_apply
    {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂}
    {Q₃ : QuadraticForm R M₃} {Q₄ : QuadraticForm R M₄}
    (f : Q₁ →qᵢ Q₂) (g : Q₃ →qᵢ Q₄) (x : M₁ ⊗[R] M₃) :
    Q₂.tmul Q₄ (TensorProduct.map f.toLinearMap g.toLinearMap x) = Q₁.tmul Q₃ x :=
  DFunLike.congr_fun (tmul_comp_tensorMap f g) x


/-- `TensorProduct.map` for `Quadraticform.Isometry`s -/
def _root_.QuadraticMap.Isometry.tmul
    {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂}
    {Q₃ : QuadraticForm R M₃} {Q₄ : QuadraticForm R M₄}
    (f : Q₁ →qᵢ Q₂) (g : Q₃ →qᵢ Q₄) : (Q₁.tmul Q₃) →qᵢ (Q₂.tmul Q₄) where
  toLinearMap := TensorProduct.map f.toLinearMap g.toLinearMap
  map_app' := tmul_tensorMap_apply f g


@[simp]
theorem _root_.QuadraticMap.Isometry.tmul_apply
    {Q₁ : QuadraticForm R M₁} {Q₂ : QuadraticForm R M₂}
    {Q₃ : QuadraticForm R M₃} {Q₄ : QuadraticForm R M₄}
    (f : Q₁ →qᵢ Q₂) (g : Q₃ →qᵢ Q₄) (x : M₁ ⊗[R] M₃) :
    f.tmul g x = TensorProduct.map f.toLinearMap g.toLinearMap x :=
  rfl


@[simp]
theorem tmul_comp_tensorComm (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) :
    (Q₂.tmul Q₁).comp (TensorProduct.comm R M₁ M₂) = Q₁.tmul Q₂ := by
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    ⊢ Eq (QuadraticMap.comp (Q₂.tmul Q₁) ↑(TensorProduct.comm R M₁ M₂)) (Q₁.tmul Q₂)
  -/
  refine (QuadraticMap.associated_rightInverse R).injective ?_
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    ⊢ Eq ((QuadraticMap.associatedHom R) (QuadraticMap.comp (Q₂.tmul Q₁) ↑(TensorP …
  -/
  ext m₁ m₂ m₁' m₂'
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    m₁ : M₁
    m₂ : M₂
    m₁' : M₁
    m₂' : M₂
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  dsimp [-associated_apply]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    m₁ : M₁
    m₂ : M₂
    m₁' : M₁
    m₂' : M₂
    ⊢ Eq ((((QuadraticMap.associatedHom R) (QuadraticMap.comp (Q₂.tmul Q₁) ↑(Tenso …
  -/
  simp only [associated_tmul, QuadraticMap.associated_comp]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    m₁ : M₁
    m₂ : M₂
    m₁' : M₁
    m₂' : M₂
    ⊢ Eq (((LinearMap.compl₁₂ (LinearMap.BilinForm.tmul (QuadraticMap.associated Q …
  -/
  exact mul_comm _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem tmul_tensorComm_apply
    (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (x : M₁ ⊗[R] M₂) :
    Q₂.tmul Q₁ (TensorProduct.comm R M₁ M₂ x) = Q₁.tmul Q₂ x :=
  DFunLike.congr_fun (tmul_comp_tensorComm Q₁ Q₂) x


/-- `TensorProduct.comm` preserves tensor products of quadratic forms. -/
@[simps toLinearEquiv]
def tensorComm (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) :
    (Q₁.tmul Q₂).IsometryEquiv (Q₂.tmul Q₁) where
  toLinearEquiv := TensorProduct.comm R M₁ M₂
  map_app' := tmul_tensorComm_apply Q₁ Q₂


@[simp] lemma tensorComm_apply (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂)
    (x : M₁ ⊗[R] M₂) :
    tensorComm Q₁ Q₂ x = TensorProduct.comm R M₁ M₂ x :=
  rfl


@[simp] lemma tensorComm_symm (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) :
    (tensorComm Q₁ Q₂).symm = tensorComm Q₂ Q₁ :=
  rfl


@[simp]
theorem tmul_comp_tensorAssoc
    (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (Q₃ : QuadraticForm R M₃) :
    (Q₁.tmul (Q₂.tmul Q₃)).comp (TensorProduct.assoc R M₁ M₂ M₃) = (Q₁.tmul Q₂).tmul Q₃ := by
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommGroup M₃
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    ⊢ Eq (QuadraticMap.comp (Q₁.tmul (Q₂.tmul Q₃)) ↑(TensorProduct.assoc R M₁ M₂ M …
  -/
  refine (QuadraticMap.associated_rightInverse R).injective ?_
  /-
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommGroup M₃
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    ⊢ Eq ((QuadraticMap.associatedHom R) (QuadraticMap.comp (Q₁.tmul (Q₂.tmul Q₃)) …
  -/
  ext m₁ m₂ m₁' m₂' m₁'' m₂''
  /-
    case a.a.h.h.h.a.a.h.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommGroup M₃
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    m₁ : M₁
    m₂ : M₂
    m₁' : M₃
    m₂' : M₁
    m₁'' : M₂
    m₂'' : M₃
    ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (TensorProduct.AlgebraTensorM …
  -/
  dsimp [-associated_apply]
  /-
    case a.a.h.h.h.a.a.h.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommGroup M₃
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    m₁ : M₁
    m₂ : M₂
    m₁' : M₃
    m₂' : M₁
    m₁'' : M₂
    m₂'' : M₃
    ⊢ Eq ((((QuadraticMap.associatedHom R) (QuadraticMap.comp (Q₁.tmul (Q₂.tmul Q₃ …
  -/
  simp only [associated_tmul, QuadraticMap.associated_comp]
  /-
    case a.a.h.h.h.a.a.h.h.h
    R : Type uR
    M₁ : Type uM₁
    M₂ : Type uM₂
    M₃ : Type uM₃
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M₁
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : AddCommGroup M₃
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M₃
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Q₃ : QuadraticForm R M₃
    m₁ : M₁
    m₂ : M₂
    m₁' : M₃
    m₂' : M₁
    m₁'' : M₂
    m₂'' : M₃
    ⊢ Eq (((LinearMap.compl₁₂ (LinearMap.BilinForm.tmul (QuadraticMap.associated Q …
  -/
  exact mul_assoc _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem tmul_tensorAssoc_apply
    (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (Q₃ : QuadraticForm R M₃)
    (x : (M₁ ⊗[R] M₂) ⊗[R] M₃) :
    Q₁.tmul (Q₂.tmul Q₃) (TensorProduct.assoc R M₁ M₂ M₃ x) = (Q₁.tmul Q₂).tmul Q₃ x :=
  DFunLike.congr_fun (tmul_comp_tensorAssoc Q₁ Q₂ Q₃) x


/-- `TensorProduct.assoc` preserves tensor products of quadratic forms. -/
@[simps toLinearEquiv]
def tensorAssoc (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (Q₃ : QuadraticForm R M₃) :
    ((Q₁.tmul Q₂).tmul Q₃).IsometryEquiv (Q₁.tmul (Q₂.tmul Q₃)) where
  toLinearEquiv := TensorProduct.assoc R M₁ M₂ M₃
  map_app' := tmul_tensorAssoc_apply Q₁ Q₂ Q₃


@[simp] lemma tensorAssoc_apply
    (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (Q₃ : QuadraticForm R M₃)
    (x : (M₁ ⊗[R] M₂) ⊗[R] M₃) :
    tensorAssoc Q₁ Q₂ Q₃ x = TensorProduct.assoc R M₁ M₂ M₃ x :=
  rfl


@[simp] lemma tensorAssoc_symm_apply
    (Q₁ : QuadraticForm R M₁) (Q₂ : QuadraticForm R M₂) (Q₃ : QuadraticForm R M₃)
    (x : M₁ ⊗[R] (M₂ ⊗[R] M₃)) :
    (tensorAssoc Q₁ Q₂ Q₃).symm x = (TensorProduct.assoc R M₁ M₂ M₃).symm x :=
  rfl


theorem comp_tensorRId_eq (Q₁ : QuadraticForm R M₁) :
    Q₁.comp (TensorProduct.rid R M₁) = Q₁.tmul (sq (R := R)) := by
  /-
    R : Type uR
    M₁ : Type uM₁
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    ⊢ Eq (QuadraticMap.comp Q₁ ↑(TensorProduct.rid R M₁)) (Q₁.tmul QuadraticMap.sq)
  -/
  refine (QuadraticMap.associated_rightInverse R).injective ?_
  /-
    R : Type uR
    M₁ : Type uM₁
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    ⊢ Eq ((QuadraticMap.associatedHom R) (QuadraticMap.comp Q₁ ↑(TensorProduct.rid …
  -/
  ext m₁ m₁'
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    m₁ m₁' : M₁
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  dsimp [-associated_apply]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    m₁ m₁' : M₁
    ⊢ Eq ((((QuadraticMap.associatedHom R) (QuadraticMap.comp Q₁ ↑(TensorProduct.r …
  -/
  simp only [associated_tmul, QuadraticMap.associated_comp]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₁ : Type uM₁
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M₁
    inst✝ : Invertible 2
    Q₁ : QuadraticForm R M₁
    m₁ m₁' : M₁
    ⊢ Eq (((LinearMap.compl₁₂ ((QuadraticMap.associatedHom R) Q₁) ↑(TensorProduct. …
  -/
  simp [-associated_apply, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem tmul_tensorRId_apply
    (Q₁ : QuadraticForm R M₁) (x : M₁ ⊗[R] R) :
    Q₁ (TensorProduct.rid R M₁ x) = Q₁.tmul (sq (R := R)) x :=
  DFunLike.congr_fun (comp_tensorRId_eq Q₁) x


/-- `TensorProduct.rid` preserves tensor products of quadratic forms. -/
@[simps toLinearEquiv]
def tensorRId (Q₁ : QuadraticForm R M₁) :
    (Q₁.tmul (sq (R := R))).IsometryEquiv Q₁ where
  toLinearEquiv := TensorProduct.rid R M₁
  map_app' := tmul_tensorRId_apply Q₁


@[simp] lemma tensorRId_apply (Q₁ : QuadraticForm R M₁) (x : M₁ ⊗[R] R) :
    tensorRId Q₁ x = TensorProduct.rid R M₁ x :=
  rfl


@[simp] lemma tensorRId_symm_apply (Q₁ : QuadraticForm R M₁) (x : M₁) :
    (tensorRId Q₁).symm x = (TensorProduct.rid R M₁).symm x :=
  rfl


theorem comp_tensorLId_eq (Q₂ : QuadraticForm R M₂) :
    Q₂.comp (TensorProduct.lid R M₂) = QuadraticForm.tmul (sq (R := R)) Q₂ := by
  /-
    R : Type uR
    M₂ : Type uM₂
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₂ : QuadraticForm R M₂
    ⊢ Eq (QuadraticMap.comp Q₂ ↑(TensorProduct.lid R M₂)) (QuadraticForm.tmul Quad …
  -/
  refine (QuadraticMap.associated_rightInverse R).injective ?_
  /-
    R : Type uR
    M₂ : Type uM₂
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₂ : QuadraticForm R M₂
    ⊢ Eq ((QuadraticMap.associatedHom R) (QuadraticMap.comp Q₂ ↑(TensorProduct.lid …
  -/
  ext m₂ m₂'
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₂ : Type uM₂
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₂ : QuadraticForm R M₂
    m₂ m₂' : M₂
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  dsimp [-associated_apply]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₂ : Type uM₂
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₂ : QuadraticForm R M₂
    m₂ m₂' : M₂
    ⊢ Eq ((((QuadraticMap.associatedHom R) (QuadraticMap.comp Q₂ ↑(TensorProduct.l …
  -/
  simp only [associated_tmul, QuadraticMap.associated_comp]
  /-
    case a.h.h.a.h.h
    R : Type uR
    M₂ : Type uM₂
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₂
    inst✝ : Invertible 2
    Q₂ : QuadraticForm R M₂
    m₂ m₂' : M₂
    ⊢ Eq (((LinearMap.compl₁₂ ((QuadraticMap.associatedHom R) Q₂) ↑(TensorProduct. …
  -/
  simp [-associated_apply, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem tmul_tensorLId_apply
    (Q₂ : QuadraticForm R M₂) (x : R ⊗[R] M₂) :
    Q₂ (TensorProduct.lid R M₂ x) = QuadraticForm.tmul (sq (R := R)) Q₂ x :=
  DFunLike.congr_fun (comp_tensorLId_eq Q₂) x


/-- `TensorProduct.lid` preserves tensor products of quadratic forms. -/
@[simps toLinearEquiv]
def tensorLId (Q₂ : QuadraticForm R M₂) :
    (QuadraticForm.tmul (sq (R := R)) Q₂).IsometryEquiv Q₂ where
  toLinearEquiv := TensorProduct.lid R M₂
  map_app' := tmul_tensorLId_apply Q₂


@[simp] lemma tensorLId_apply (Q₂ : QuadraticForm R M₂) (x : R ⊗[R] M₂) :
    tensorLId Q₂ x = TensorProduct.lid R M₂ x :=
  rfl


@[simp] lemma tensorLId_symm_apply (Q₂ : QuadraticForm R M₂) (x : M₂) :
    (tensorLId Q₂).symm x = (TensorProduct.lid R M₂).symm x :=
  rfl


