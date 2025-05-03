@[simp, norm_cast]
theorem ofReal_prod (f : α → ℝ) : ((∏ i ∈ s, f i : ℝ) : ℂ) = ∏ i ∈ s, (f i : ℂ) :=
  map_prod ofRealHom _ _


@[simp, norm_cast]
theorem ofReal_sum (f : α → ℝ) : ((∑ i ∈ s, f i : ℝ) : ℂ) = ∑ i ∈ s, (f i : ℂ) :=
  map_sum ofRealHom _ _


@[simp, norm_cast]
lemma ofReal_expect (f : α → ℝ) : (𝔼 i ∈ s, f i : ℝ) = 𝔼 i ∈ s, (f i : ℂ) :=
  map_expect ofRealHom ..


@[simp, norm_cast]
lemma ofReal_balance [Fintype α] (f : α → ℝ) (a : α) :
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : Fintype α
                                                          f : α → Real
                                                          a : α
                                                          ⊢ Eq (↑(Fintype.balance f a)) (Fintype.balance (Function.comp Complex.ofReal f …
                                                        -/
    ((balance f a : ℝ) : ℂ) = balance ((↑) ∘ f) a := by simp [balance]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] lemma ofReal_comp_balance {ι : Type*} [Fintype ι] (f : ι → ℝ) :
    ofReal ∘ balance f = balance (ofReal ∘ f : ι → ℂ) := funext <| ofReal_balance _


@[simp]
theorem re_sum (f : α → ℂ) : (∑ i ∈ s, f i).re = ∑ i ∈ s, (f i).re :=
  map_sum reAddGroupHom f s


@[simp]
lemma re_expect (f : α → ℂ) : (𝔼 i ∈ s, f i).re = 𝔼 i ∈ s, (f i).re :=
                                                      /-
                                                        α : Type u_1
                                                        s : Finset α
                                                        f : α → Complex
                                                        ⊢ ∀ (m : NNRat) (x : Complex), Eq ((↑Complex.reAddGroupHom).toFun (HSMul.hSMul …
                                                      -/
  map_expect (LinearMap.mk reAddGroupHom.toAddHom (by simp)) f s
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
lemma re_balance [Fintype α] (f : α → ℂ) (a : α) : re (balance f a) = balance (re ∘ f) a := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    f : α → Complex
    a : α
    ⊢ Eq (Fintype.balance f a).re (Fintype.balance (Function.comp Complex.re f) a)
  -/
  simp [balance]
  /-
    🎉 no goals
  -/


@[simp] lemma re_comp_balance {ι : Type*} [Fintype ι] (f : ι → ℂ) :
    re ∘ balance f = balance (re ∘ f) := funext <| re_balance _


@[simp]
theorem im_sum (f : α → ℂ) : (∑ i ∈ s, f i).im = ∑ i ∈ s, (f i).im :=
  map_sum imAddGroupHom f s


@[simp]
lemma im_expect (f : α → ℂ) : (𝔼 i ∈ s, f i).im = 𝔼 i ∈ s, (f i).im :=
                                                      /-
                                                        α : Type u_1
                                                        s : Finset α
                                                        f : α → Complex
                                                        ⊢ ∀ (m : NNRat) (x : Complex), Eq ((↑Complex.imAddGroupHom).toFun (HSMul.hSMul …
                                                      -/
  map_expect (LinearMap.mk imAddGroupHom.toAddHom (by simp)) f s
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
lemma im_balance [Fintype α] (f : α → ℂ) (a : α) : im (balance f a) = balance (im ∘ f) a := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    f : α → Complex
    a : α
    ⊢ Eq (Fintype.balance f a).im (Fintype.balance (Function.comp Complex.im f) a)
  -/
  simp [balance]
  /-
    🎉 no goals
  -/


@[simp] lemma im_comp_balance {ι : Type*} [Fintype ι] (f : ι → ℂ) :
    im ∘ balance f = balance (im ∘ f) := funext <| im_balance _


