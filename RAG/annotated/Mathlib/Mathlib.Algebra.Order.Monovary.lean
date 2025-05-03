@[to_additive (attr := simp)]
lemma monovaryOn_inv_left : MonovaryOn f⁻¹ g s ↔ AntivaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    s : Set ι
    f : ι → α
    g : ι → β
    ⊢ Iff (MonovaryOn (Inv.inv f) g s) (AntivaryOn f g s)
  -/
  simp [MonovaryOn, AntivaryOn]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma antivaryOn_inv_left : AntivaryOn f⁻¹ g s ↔ MonovaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    s : Set ι
    f : ι → α
    g : ι → β
    ⊢ Iff (AntivaryOn (Inv.inv f) g s) (MonovaryOn f g s)
  -/
  simp [MonovaryOn, AntivaryOn]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma monovaryOn_inv_right : MonovaryOn f g⁻¹ s ↔ AntivaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    s : Set ι
    f : ι → α
    g : ι → β
    ⊢ Iff (MonovaryOn f (Inv.inv g) s) (AntivaryOn f g s)
  -/
  simpa [MonovaryOn, AntivaryOn] using forall₂_swap
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma antivaryOn_inv_right : AntivaryOn f g⁻¹ s ↔ MonovaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    s : Set ι
    f : ι → α
    g : ι → β
    ⊢ Iff (AntivaryOn f (Inv.inv g) s) (MonovaryOn f g s)
  -/
  simpa [MonovaryOn, AntivaryOn] using forall₂_swap
  /-
    🎉 no goals
  -/


                                                                                    /-
                                                                                      ι : Type u_1
                                                                                      α : Type u_2
                                                                                      β : Type u_3
                                                                                      inst✝¹ : OrderedCommGroup α
                                                                                      inst✝ : OrderedCommGroup β
                                                                                      s : Set ι
                                                                                      f : ι → α
                                                                                      g : ι → β
                                                                                      ⊢ Iff (MonovaryOn (Inv.inv f) (Inv.inv g) s) (MonovaryOn f g s)
                                                                                    -/
@[to_additive] lemma monovaryOn_inv : MonovaryOn f⁻¹ g⁻¹ s ↔ MonovaryOn f g s := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

                                                                                    /-
                                                                                      ι : Type u_1
                                                                                      α : Type u_2
                                                                                      β : Type u_3
                                                                                      inst✝¹ : OrderedCommGroup α
                                                                                      inst✝ : OrderedCommGroup β
                                                                                      s : Set ι
                                                                                      f : ι → α
                                                                                      g : ι → β
                                                                                      ⊢ Iff (AntivaryOn (Inv.inv f) (Inv.inv g) s) (AntivaryOn f g s)
                                                                                    -/
@[to_additive] lemma antivaryOn_inv : AntivaryOn f⁻¹ g⁻¹ s ↔ AntivaryOn f g s := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[to_additive (attr := simp)] lemma monovary_inv_left : Monovary f⁻¹ g ↔ Antivary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    f : ι → α
    g : ι → β
    ⊢ Iff (Monovary (Inv.inv f) g) (Antivary f g)
  -/
  simp [Monovary, Antivary]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)] lemma antivary_inv_left : Antivary f⁻¹ g ↔ Monovary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    f : ι → α
    g : ι → β
    ⊢ Iff (Antivary (Inv.inv f) g) (Monovary f g)
  -/
  simp [Monovary, Antivary]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)] lemma monovary_inv_right : Monovary f g⁻¹ ↔ Antivary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    f : ι → α
    g : ι → β
    ⊢ Iff (Monovary f (Inv.inv g)) (Antivary f g)
  -/
  simpa [Monovary, Antivary] using forall_swap
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)] lemma antivary_inv_right : Antivary f g⁻¹ ↔ Monovary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OrderedCommGroup α
    inst✝ : OrderedCommGroup β
    f : ι → α
    g : ι → β
    ⊢ Iff (Antivary f (Inv.inv g)) (Monovary f g)
  -/
  simpa [Monovary, Antivary] using forall_swap
  /-
    🎉 no goals
  -/


                                                                          /-
                                                                            ι : Type u_1
                                                                            α : Type u_2
                                                                            β : Type u_3
                                                                            inst✝¹ : OrderedCommGroup α
                                                                            inst✝ : OrderedCommGroup β
                                                                            f : ι → α
                                                                            g : ι → β
                                                                            ⊢ Iff (Monovary (Inv.inv f) (Inv.inv g)) (Monovary f g)
                                                                          -/
@[to_additive] lemma monovary_inv : Monovary f⁻¹ g⁻¹ ↔ Monovary f g := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                          /-
                                                                            ι : Type u_1
                                                                            α : Type u_2
                                                                            β : Type u_3
                                                                            inst✝¹ : OrderedCommGroup α
                                                                            inst✝ : OrderedCommGroup β
                                                                            f : ι → α
                                                                            g : ι → β
                                                                            ⊢ Iff (Antivary (Inv.inv f) (Inv.inv g)) (Antivary f g)
                                                                          -/
@[to_additive] lemma antivary_inv : Antivary f⁻¹ g⁻¹ ↔ Antivary f g := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive] alias ⟨MonovaryOn.of_inv_left, AntivaryOn.inv_left⟩ := monovaryOn_inv_left

@[to_additive] alias ⟨AntivaryOn.of_inv_left, MonovaryOn.inv_left⟩ := antivaryOn_inv_left

@[to_additive] alias ⟨MonovaryOn.of_inv_right, AntivaryOn.inv_right⟩ := monovaryOn_inv_right

@[to_additive] alias ⟨AntivaryOn.of_inv_right, MonovaryOn.inv_right⟩ := antivaryOn_inv_right

@[to_additive] alias ⟨MonovaryOn.of_inv, MonovaryOn.inv⟩ := monovaryOn_inv

@[to_additive] alias ⟨AntivaryOn.of_inv, AntivaryOn.inv⟩ := antivaryOn_inv

@[to_additive] alias ⟨Monovary.of_inv_left, Antivary.inv_left⟩ := monovary_inv_left

@[to_additive] alias ⟨Antivary.of_inv_left, Monovary.inv_left⟩ := antivary_inv_left

@[to_additive] alias ⟨Monovary.of_inv_right, Antivary.inv_right⟩ := monovary_inv_right

@[to_additive] alias ⟨Antivary.of_inv_right, Monovary.inv_right⟩ := antivary_inv_right

@[to_additive] alias ⟨Monovary.of_inv, Monovary.inv⟩ := monovary_inv

@[to_additive] alias ⟨Antivary.of_inv, Antivary.inv⟩ := antivary_inv


@[to_additive] lemma MonovaryOn.mul_left (h₁ : MonovaryOn f₁ g s) (h₂ : MonovaryOn f₂ g s) :
    MonovaryOn (f₁ * f₂) g s := fun _i hi _j hj hij ↦ mul_le_mul' (h₁ hi hj hij) (h₂ hi hj hij)


@[to_additive] lemma AntivaryOn.mul_left (h₁ : AntivaryOn f₁ g s) (h₂ : AntivaryOn f₂ g s) :
    AntivaryOn (f₁ * f₂) g s := fun _i hi _j hj hij ↦ mul_le_mul' (h₁ hi hj hij) (h₂ hi hj hij)


@[to_additive] lemma MonovaryOn.div_left (h₁ : MonovaryOn f₁ g s) (h₂ : AntivaryOn f₂ g s) :
    MonovaryOn (f₁ / f₂) g s := fun _i hi _j hj hij ↦ div_le_div'' (h₁ hi hj hij) (h₂ hi hj hij)


@[to_additive] lemma AntivaryOn.div_left (h₁ : AntivaryOn f₁ g s) (h₂ : MonovaryOn f₂ g s) :
    AntivaryOn (f₁ / f₂) g s := fun _i hi _j hj hij ↦ div_le_div'' (h₁ hi hj hij) (h₂ hi hj hij)


@[to_additive] lemma MonovaryOn.pow_left (hfg : MonovaryOn f g s) (n : ℕ) :
    MonovaryOn (f ^ n) g s := fun _i hi _j hj hij ↦ pow_le_pow_left' (hfg hi hj hij) _


@[to_additive] lemma AntivaryOn.pow_left (hfg : AntivaryOn f g s) (n : ℕ) :
    AntivaryOn (f ^ n) g s := fun _i hi _j hj hij ↦ pow_le_pow_left' (hfg hi hj hij) _


@[to_additive]
lemma Monovary.mul_left (h₁ : Monovary f₁ g) (h₂ : Monovary f₂ g) : Monovary (f₁ * f₂) g :=
  fun _i _j hij ↦ mul_le_mul' (h₁ hij) (h₂ hij)


@[to_additive]
lemma Antivary.mul_left (h₁ : Antivary f₁ g) (h₂ : Antivary f₂ g) : Antivary (f₁ * f₂) g :=
  fun _i _j hij ↦ mul_le_mul' (h₁ hij) (h₂ hij)


@[to_additive]
lemma Monovary.div_left (h₁ : Monovary f₁ g) (h₂ : Antivary f₂ g) : Monovary (f₁ / f₂) g :=
  fun _i _j hij ↦ div_le_div'' (h₁ hij) (h₂ hij)


@[to_additive]
lemma Antivary.div_left (h₁ : Antivary f₁ g) (h₂ : Monovary f₂ g) : Antivary (f₁ / f₂) g :=
  fun _i _j hij ↦ div_le_div'' (h₁ hij) (h₂ hij)


@[to_additive] lemma Monovary.pow_left (hfg : Monovary f g) (n : ℕ) : Monovary (f ^ n) g :=
  fun _i _j hij ↦ pow_le_pow_left' (hfg hij) _


@[to_additive] lemma Antivary.pow_left (hfg : Antivary f g) (n : ℕ) : Antivary (f ^ n) g :=
  fun _i _j hij ↦ pow_le_pow_left' (hfg hij) _


@[to_additive] lemma MonovaryOn.mul_right (h₁ : MonovaryOn f g₁ s) (h₂ : MonovaryOn f g₂ s) :
    MonovaryOn f (g₁ * g₂) s :=
  fun _i hi _j hj hij ↦ (lt_or_lt_of_mul_lt_mul hij).elim (h₁ hi hj) <| h₂ hi hj


@[to_additive] lemma AntivaryOn.mul_right (h₁ : AntivaryOn f g₁ s) (h₂ : AntivaryOn f g₂ s) :
    AntivaryOn f (g₁ * g₂) s :=
  fun _i hi _j hj hij ↦ (lt_or_lt_of_mul_lt_mul hij).elim (h₁ hi hj) <| h₂ hi hj


@[to_additive] lemma MonovaryOn.div_right (h₁ : MonovaryOn f g₁ s) (h₂ : AntivaryOn f g₂ s) :
    MonovaryOn f (g₁ / g₂) s :=
  fun _i hi _j hj hij ↦ (lt_or_lt_of_div_lt_div hij).elim (h₁ hi hj) <| h₂ hj hi


@[to_additive] lemma AntivaryOn.div_right (h₁ : AntivaryOn f g₁ s) (h₂ : MonovaryOn f g₂ s) :
    AntivaryOn f (g₁ / g₂) s :=
  fun _i hi _j hj hij ↦ (lt_or_lt_of_div_lt_div hij).elim (h₁ hi hj) <| h₂ hj hi


@[to_additive] lemma MonovaryOn.pow_right (hfg : MonovaryOn f g s) (n : ℕ) :
    MonovaryOn f (g ^ n) s := fun _i hi _j hj hij ↦ hfg hi hj <| lt_of_pow_lt_pow_left' _ hij


@[to_additive] lemma AntivaryOn.pow_right (hfg : AntivaryOn f g s) (n : ℕ) :
    AntivaryOn f (g ^ n) s := fun _i hi _j hj hij ↦ hfg hi hj <| lt_of_pow_lt_pow_left' _ hij


@[to_additive] lemma Monovary.mul_right (h₁ : Monovary f g₁) (h₂ : Monovary f g₂) :
    Monovary f (g₁ * g₂) :=
  fun _i _j hij ↦ (lt_or_lt_of_mul_lt_mul hij).elim (fun h ↦ h₁ h) fun h ↦ h₂ h


@[to_additive] lemma Antivary.mul_right (h₁ : Antivary f g₁) (h₂ : Antivary f g₂) :
    Antivary f (g₁ * g₂) :=
  fun _i _j hij ↦ (lt_or_lt_of_mul_lt_mul hij).elim (fun h ↦ h₁ h) fun h ↦ h₂ h


@[to_additive] lemma Monovary.div_right (h₁ : Monovary f g₁) (h₂ : Antivary f g₂) :
    Monovary f (g₁ / g₂) :=
  fun _i _j hij ↦ (lt_or_lt_of_div_lt_div hij).elim (fun h ↦ h₁ h) fun h ↦ h₂ h


@[to_additive] lemma Antivary.div_right (h₁ : Antivary f g₁) (h₂ : Monovary f g₂) :
    Antivary f (g₁ / g₂) :=
  fun _i _j hij ↦ (lt_or_lt_of_div_lt_div hij).elim (fun h ↦ h₁ h) fun h ↦ h₂ h


@[to_additive] lemma Monovary.pow_right (hfg : Monovary f g) (n : ℕ) : Monovary f (g ^ n) :=
  fun _i _j hij ↦ hfg <| lt_of_pow_lt_pow_left' _ hij


@[to_additive] lemma Antivary.pow_right (hfg : Antivary f g) (n : ℕ) : Antivary f (g ^ n) :=
  fun _i _j hij ↦ hfg <| lt_of_pow_lt_pow_left' _ hij


lemma MonovaryOn.mul_left₀ (hf₁ : ∀ i ∈ s, 0 ≤ f₁ i) (hf₂ : ∀ i ∈ s, 0 ≤ f₂ i)
    (h₁ : MonovaryOn f₁ g s) (h₂ : MonovaryOn f₂ g s) : MonovaryOn (f₁ * f₂) g s :=
  fun _i hi _j hj hij ↦ mul_le_mul (h₁ hi hj hij) (h₂ hi hj hij) (hf₂ _ hi) (hf₁ _ hj)


lemma AntivaryOn.mul_left₀ (hf₁ : ∀ i ∈ s, 0 ≤ f₁ i) (hf₂ : ∀ i ∈ s, 0 ≤ f₂ i)
    (h₁ : AntivaryOn f₁ g s) (h₂ : AntivaryOn f₂ g s) : AntivaryOn (f₁ * f₂) g s :=
  fun _i hi _j hj hij ↦ mul_le_mul (h₁ hi hj hij) (h₂ hi hj hij) (hf₂ _ hj) (hf₁ _ hi)


lemma MonovaryOn.pow_left₀ (hf : ∀ i ∈ s, 0 ≤ f i) (hfg : MonovaryOn f g s) (n : ℕ) :
    MonovaryOn (f ^ n) g s :=
  fun _i hi _j hj hij ↦ pow_le_pow_left₀ (hf _ hi) (hfg hi hj hij) _


lemma AntivaryOn.pow_left₀ (hf : ∀ i ∈ s, 0 ≤ f i) (hfg : AntivaryOn f g s) (n : ℕ) :
    AntivaryOn (f ^ n) g s :=
  fun _i hi _j hj hij ↦ pow_le_pow_left₀ (hf _ hj) (hfg hi hj hij) _


lemma Monovary.mul_left₀ (hf₁ : 0 ≤ f₁) (hf₂ : 0 ≤ f₂) (h₁ : Monovary f₁ g) (h₂ : Monovary f₂ g) :
    Monovary (f₁ * f₂) g := fun _i _j hij ↦ mul_le_mul (h₁ hij) (h₂ hij) (hf₂ _) (hf₁ _)


lemma Antivary.mul_left₀ (hf₁ : 0 ≤ f₁) (hf₂ : 0 ≤ f₂) (h₁ : Antivary f₁ g) (h₂ : Antivary f₂ g) :
    Antivary (f₁ * f₂) g := fun _i _j hij ↦ mul_le_mul (h₁ hij) (h₂ hij) (hf₂ _) (hf₁ _)


lemma Monovary.pow_left₀ (hf : 0 ≤ f) (hfg : Monovary f g) (n : ℕ) : Monovary (f ^ n) g :=
  fun _i _j hij ↦ pow_le_pow_left₀ (hf _) (hfg hij) _


lemma Antivary.pow_left₀ (hf : 0 ≤ f) (hfg : Antivary f g) (n : ℕ) : Antivary (f ^ n) g :=
  fun _i _j hij ↦ pow_le_pow_left₀ (hf _) (hfg hij) _


lemma MonovaryOn.mul_right₀ (hg₁ : ∀ i ∈ s, 0 ≤ g₁ i) (hg₂ : ∀ i ∈ s, 0 ≤ g₂ i)
    (h₁ : MonovaryOn f g₁ s) (h₂ : MonovaryOn f g₂ s) : MonovaryOn f (g₁ * g₂) s :=
  (h₁.symm.mul_left₀ hg₁ hg₂ h₂.symm).symm


lemma AntivaryOn.mul_right₀ (hg₁ : ∀ i ∈ s, 0 ≤ g₁ i) (hg₂ : ∀ i ∈ s, 0 ≤ g₂ i)
    (h₁ : AntivaryOn f g₁ s) (h₂ : AntivaryOn f g₂ s) : AntivaryOn f (g₁ * g₂) s :=
  (h₁.symm.mul_left₀ hg₁ hg₂ h₂.symm).symm


lemma MonovaryOn.pow_right₀ (hg : ∀ i ∈ s, 0 ≤ g i) (hfg : MonovaryOn f g s) (n : ℕ) :
    MonovaryOn f (g ^ n) s := (hfg.symm.pow_left₀ hg _).symm


lemma AntivaryOn.pow_right₀ (hg : ∀ i ∈ s, 0 ≤ g i) (hfg : AntivaryOn f g s) (n : ℕ) :
    AntivaryOn f (g ^ n) s := (hfg.symm.pow_left₀ hg _).symm


lemma Monovary.mul_right₀ (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (h₁ : Monovary f g₁) (h₂ : Monovary f g₂) :
    Monovary f (g₁ * g₂) := (h₁.symm.mul_left₀ hg₁ hg₂ h₂.symm).symm


lemma Antivary.mul_right₀ (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (h₁ : Antivary f g₁) (h₂ : Antivary f g₂) :
    Antivary f (g₁ * g₂) := (h₁.symm.mul_left₀ hg₁ hg₂ h₂.symm).symm


lemma Monovary.pow_right₀ (hg : 0 ≤ g) (hfg : Monovary f g) (n : ℕ) : Monovary f (g ^ n) :=
  (hfg.symm.pow_left₀ hg _).symm


lemma Antivary.pow_right₀ (hg : 0 ≤ g) (hfg : Antivary f g) (n : ℕ) : Antivary f (g ^ n) :=
  (hfg.symm.pow_left₀ hg _).symm


@[simp]
lemma monovaryOn_inv_left₀ (hf : ∀ i ∈ s, 0 < f i) : MonovaryOn f⁻¹ g s ↔ AntivaryOn f g s :=
  forall₅_congr fun _i hi _j hj _ ↦ inv_le_inv₀ (hf _ hi) (hf _ hj)


@[simp]
lemma antivaryOn_inv_left₀ (hf : ∀ i ∈ s, 0 < f i) : AntivaryOn f⁻¹ g s ↔ MonovaryOn f g s :=
  forall₅_congr fun _i hi _j hj _ ↦ inv_le_inv₀ (hf _ hj) (hf _ hi)


@[simp]
lemma monovaryOn_inv_right₀ (hg : ∀ i ∈ s, 0 < g i) : MonovaryOn f g⁻¹ s ↔ AntivaryOn f g s :=
                                                         /-
                                                           ι : Type u_1
                                                           α : Type u_2
                                                           β : Type u_3
                                                           inst✝¹ : LinearOrderedSemifield α
                                                           inst✝ : LinearOrderedSemifield β
                                                           s : Set ι
                                                           f : ι → α
                                                           g : ι → β
                                                           hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
                                                           i : ι
                                                           hi : Membership.mem s i
                                                           j : ι
                                                           hj : Membership.mem s j
                                                           ⊢ Iff (LT.lt (Inv.inv g j) (Inv.inv g i) → LE.le (f j) (f i)) (LT.lt (g i) (g  …
                                                         -/
  forall₂_swap.trans <| forall₄_congr fun i hi j hj ↦ by erw [inv_lt_inv₀ (hg _ hj) (hg _ hi)]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
lemma antivaryOn_inv_right₀ (hg : ∀ i ∈ s, 0 < g i) : AntivaryOn f g⁻¹ s ↔ MonovaryOn f g s :=
                                                         /-
                                                           ι : Type u_1
                                                           α : Type u_2
                                                           β : Type u_3
                                                           inst✝¹ : LinearOrderedSemifield α
                                                           inst✝ : LinearOrderedSemifield β
                                                           s : Set ι
                                                           f : ι → α
                                                           g : ι → β
                                                           hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
                                                           i : ι
                                                           hi : Membership.mem s i
                                                           j : ι
                                                           hj : Membership.mem s j
                                                           ⊢ Iff (LT.lt (Inv.inv g j) (Inv.inv g i) → LE.le (f i) (f j)) (LT.lt (g i) (g  …
                                                         -/
  forall₂_swap.trans <| forall₄_congr fun i hi j hj ↦ by erw [inv_lt_inv₀ (hg _ hj) (hg _ hi)]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma monovaryOn_inv₀ (hf : ∀ i ∈ s, 0 < f i) (hg : ∀ i ∈ s, 0 < g i) :
    MonovaryOn f⁻¹ g⁻¹ s ↔ MonovaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : LinearOrderedSemifield β
    s : Set ι
    f : ι → α
    g : ι → β
    hf : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    ⊢ Iff (MonovaryOn (Inv.inv f) (Inv.inv g) s) (MonovaryOn f g s)
  -/
  rw [monovaryOn_inv_left₀ hf, antivaryOn_inv_right₀ hg]
  /-
    🎉 no goals
  -/

lemma antivaryOn_inv₀ (hf : ∀ i ∈ s, 0 < f i) (hg : ∀ i ∈ s, 0 < g i) :
    AntivaryOn f⁻¹ g⁻¹ s ↔ AntivaryOn f g s := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : LinearOrderedSemifield β
    s : Set ι
    f : ι → α
    g : ι → β
    hf : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    ⊢ Iff (AntivaryOn (Inv.inv f) (Inv.inv g) s) (AntivaryOn f g s)
  -/
  rw [antivaryOn_inv_left₀ hf, monovaryOn_inv_right₀ hg]
  /-
    🎉 no goals
  -/


@[simp] lemma monovary_inv_left₀ (hf : StrongLT 0 f) : Monovary f⁻¹ g ↔ Antivary f g :=
  forall₃_congr fun _i _j _ ↦ inv_le_inv₀ (hf _) (hf _)


@[simp] lemma antivary_inv_left₀ (hf : StrongLT 0 f) : Antivary f⁻¹ g ↔ Monovary f g :=
  forall₃_congr fun _i _j _ ↦ inv_le_inv₀ (hf _) (hf _)


@[simp] lemma monovary_inv_right₀ (hg : StrongLT 0 g) : Monovary f g⁻¹ ↔ Antivary f g :=
                                                  /-
                                                    ι : Type u_1
                                                    α : Type u_2
                                                    β : Type u_3
                                                    inst✝¹ : LinearOrderedSemifield α
                                                    inst✝ : LinearOrderedSemifield β
                                                    f : ι → α
                                                    g : ι → β
                                                    hg : StrongLT 0 g
                                                    i j : ι
                                                    ⊢ Iff (LT.lt (Inv.inv g j) (Inv.inv g i) → LE.le (f j) (f i)) (LT.lt (g i) (g  …
                                                  -/
  forall_swap.trans <| forall₂_congr fun i j ↦ by erw [inv_lt_inv₀ (hg _) (hg _)]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] lemma antivary_inv_right₀ (hg : StrongLT 0 g) : Antivary f g⁻¹ ↔ Monovary f g :=
                                                  /-
                                                    ι : Type u_1
                                                    α : Type u_2
                                                    β : Type u_3
                                                    inst✝¹ : LinearOrderedSemifield α
                                                    inst✝ : LinearOrderedSemifield β
                                                    f : ι → α
                                                    g : ι → β
                                                    hg : StrongLT 0 g
                                                    i j : ι
                                                    ⊢ Iff (LT.lt (Inv.inv g j) (Inv.inv g i) → LE.le (f i) (f j)) (LT.lt (g i) (g  …
                                                  -/
  forall_swap.trans <| forall₂_congr fun i j ↦ by erw [inv_lt_inv₀ (hg _) (hg _)]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma monovary_inv₀ (hf : StrongLT 0 f) (hg : StrongLT 0 g) : Monovary f⁻¹ g⁻¹ ↔ Monovary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : LinearOrderedSemifield β
    f : ι → α
    g : ι → β
    hf : StrongLT 0 f
    hg : StrongLT 0 g
    ⊢ Iff (Monovary (Inv.inv f) (Inv.inv g)) (Monovary f g)
  -/
  rw [monovary_inv_left₀ hf, antivary_inv_right₀ hg]
  /-
    🎉 no goals
  -/

lemma antivary_inv₀ (hf : StrongLT 0 f) (hg : StrongLT 0 g) : Antivary f⁻¹ g⁻¹ ↔ Antivary f g := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : LinearOrderedSemifield β
    f : ι → α
    g : ι → β
    hf : StrongLT 0 f
    hg : StrongLT 0 g
    ⊢ Iff (Antivary (Inv.inv f) (Inv.inv g)) (Antivary f g)
  -/
  rw [antivary_inv_left₀ hf, monovary_inv_right₀ hg]
  /-
    🎉 no goals
  -/


alias ⟨MonovaryOn.of_inv_left₀, AntivaryOn.inv_left₀⟩ := monovaryOn_inv_left₀

alias ⟨AntivaryOn.of_inv_left₀, MonovaryOn.inv_left₀⟩ := antivaryOn_inv_left₀

alias ⟨MonovaryOn.of_inv_right₀, AntivaryOn.inv_right₀⟩ := monovaryOn_inv_right₀

alias ⟨AntivaryOn.of_inv_right₀, MonovaryOn.inv_right₀⟩ := antivaryOn_inv_right₀

alias ⟨MonovaryOn.of_inv₀, MonovaryOn.inv₀⟩ := monovaryOn_inv₀

alias ⟨AntivaryOn.of_inv₀, AntivaryOn.inv₀⟩ := antivaryOn_inv₀

alias ⟨Monovary.of_inv_left₀, Antivary.inv_left₀⟩ := monovary_inv_left₀

alias ⟨Antivary.of_inv_left₀, Monovary.inv_left₀⟩ := antivary_inv_left₀

alias ⟨Monovary.of_inv_right₀, Antivary.inv_right₀⟩ := monovary_inv_right₀

alias ⟨Antivary.of_inv_right₀, Monovary.inv_right₀⟩ := antivary_inv_right₀

alias ⟨Monovary.of_inv₀, Monovary.inv₀⟩ := monovary_inv₀

alias ⟨Antivary.of_inv₀, Antivary.inv₀⟩ := antivary_inv₀


lemma MonovaryOn.div_left₀ (hf₁ : ∀ i ∈ s, 0 ≤ f₁ i) (hf₂ : ∀ i ∈ s, 0 < f₂ i)
    (h₁ : MonovaryOn f₁ g s) (h₂ : AntivaryOn f₂ g s) : MonovaryOn (f₁ / f₂) g s :=
  fun _i hi _j hj hij ↦ div_le_div₀ (hf₁ _ hj) (h₁ hi hj hij) (hf₂ _ hj) <| h₂ hi hj hij


lemma AntivaryOn.div_left₀ (hf₁ : ∀ i ∈ s, 0 ≤ f₁ i) (hf₂ : ∀ i ∈ s, 0 < f₂ i)
    (h₁ : AntivaryOn f₁ g s) (h₂ : MonovaryOn f₂ g s) : AntivaryOn (f₁ / f₂) g s :=
  fun _i hi _j hj hij ↦ div_le_div₀ (hf₁ _ hi) (h₁ hi hj hij) (hf₂ _ hi) <| h₂ hi hj hij


lemma Monovary.div_left₀ (hf₁ : 0 ≤ f₁) (hf₂ : StrongLT 0 f₂) (h₁ : Monovary f₁ g)
    (h₂ : Antivary f₂ g) : Monovary (f₁ / f₂) g :=
  fun _i _j hij ↦ div_le_div₀ (hf₁ _) (h₁ hij) (hf₂ _) <| h₂ hij


lemma Antivary.div_left₀ (hf₁ : 0 ≤ f₁) (hf₂ : StrongLT 0 f₂) (h₁ : Antivary f₁ g)
    (h₂ : Monovary f₂ g) : Antivary (f₁ / f₂) g :=
  fun _i _j hij ↦ div_le_div₀ (hf₁ _) (h₁ hij) (hf₂ _) <| h₂ hij


lemma MonovaryOn.div_right₀ (hg₁ : ∀ i ∈ s, 0 ≤ g₁ i) (hg₂ : ∀ i ∈ s, 0 < g₂ i)
    (h₁ : MonovaryOn f g₁ s) (h₂ : AntivaryOn f g₂ s) : MonovaryOn f (g₁ / g₂) s :=
  (h₁.symm.div_left₀ hg₁ hg₂ h₂.symm).symm


lemma AntivaryOn.div_right₀ (hg₁ : ∀ i ∈ s, 0 ≤ g₁ i) (hg₂ : ∀ i ∈ s, 0 < g₂ i)
    (h₁ : AntivaryOn f g₁ s) (h₂ : MonovaryOn f g₂ s) : AntivaryOn f (g₁ / g₂) s :=
  (h₁.symm.div_left₀ hg₁ hg₂ h₂.symm).symm


lemma Monovary.div_right₀ (hg₁ : 0 ≤ g₁) (hg₂ : StrongLT 0 g₂) (h₁ : Monovary f g₁)
    (h₂ : Antivary f g₂) : Monovary f (g₁ / g₂) := (h₁.symm.div_left₀ hg₁ hg₂ h₂.symm).symm


lemma Antivary.div_right₀ (hg₁ : 0 ≤ g₁) (hg₂ : StrongLT 0 g₂) (h₁ : Antivary f g₁)
    (h₂ : Monovary f g₂) : Antivary f (g₁ / g₂) := (h₁.symm.div_left₀ hg₁ hg₂ h₂.symm).symm


lemma monovaryOn_iff_forall_smul_nonneg :
    MonovaryOn f g s ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → 0 ≤ (f j - f i) • (g j - g i) := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : OrderedSMul α β
    f : ι → α
    g : ι → β
    s : Set ι
    ⊢ Iff (MonovaryOn f g s) (∀ ⦃i : ι⦄, Membership.mem s i → ∀ ⦃j : ι⦄, Membershi …
  -/
  simp_rw [smul_nonneg_iff_pos_imp_nonneg, sub_pos, sub_nonneg, forall_and]
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : LinearOrderedRing α
    inst✝² : LinearOrderedAddCommGroup β
    inst✝¹ : Module α β
    inst✝ : OrderedSMul α β
    f : ι → α
    g : ι → β
    s : Set ι
    ⊢ Iff (MonovaryOn f g s) (And (∀ (x : ι), Membership.mem s x → ∀ (x_2 : ι), Me …
  -/
  exact (and_iff_right_of_imp MonovaryOn.symm).symm
  /-
    🎉 no goals
  -/


lemma antivaryOn_iff_forall_smul_nonpos :
    AntivaryOn f g s ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → (f j - f i) • (g j - g i) ≤ 0 :=
                                           /-
                                             ι : Type u_1
                                             α : Type u_2
                                             β : Type u_3
                                             inst✝³ : LinearOrderedRing α
                                             inst✝² : LinearOrderedAddCommGroup β
                                             inst✝¹ : Module α β
                                             inst✝ : OrderedSMul α β
                                             f : ι → α
                                             g : ι → β
                                             s : Set ι
                                             ⊢ Iff (MonovaryOn f (Function.comp (⇑OrderDual.toDual) g) s) (∀ ⦃i : ι⦄, Membe …
                                           -/
  monovaryOn_toDual_right.symm.trans <| by rw [monovaryOn_iff_forall_smul_nonneg]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma monovary_iff_forall_smul_nonneg : Monovary f g ↔ ∀ i j, 0 ≤ (f j - f i) • (g j - g i) :=
  monovaryOn_univ.symm.trans <| monovaryOn_iff_forall_smul_nonneg.trans <| by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : LinearOrderedRing α
      inst✝² : LinearOrderedAddCommGroup β
      inst✝¹ : Module α β
      inst✝ : OrderedSMul α β
      f : ι → α
      g : ι → β
      ⊢ Iff (∀ ⦃i : ι⦄, Membership.mem Set.univ i → ∀ ⦃j : ι⦄, Membership.mem Set.un …
    -/
    simp only [Set.mem_univ, forall_true_left]
    /-
      🎉 no goals
    -/


lemma antivary_iff_forall_smul_nonpos : Antivary f g ↔ ∀ i j, (f j - f i) • (g j - g i) ≤ 0 :=
                                       /-
                                         ι : Type u_1
                                         α : Type u_2
                                         β : Type u_3
                                         inst✝³ : LinearOrderedRing α
                                         inst✝² : LinearOrderedAddCommGroup β
                                         inst✝¹ : Module α β
                                         inst✝ : OrderedSMul α β
                                         f : ι → α
                                         g : ι → β
                                         ⊢ Iff (Monovary f (Function.comp (⇑OrderDual.toDual) g)) (∀ (i j : ι), LE.le ( …
                                       -/
monovary_toDual_right.symm.trans <| by rw [monovary_iff_forall_smul_nonneg]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Two functions monovary iff the rearrangement inequality holds. -/
lemma monovaryOn_iff_smul_rearrangement :
    MonovaryOn f g s ↔
      ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → f i • g j + f j • g i ≤ f i • g i + f j • g j :=
  monovaryOn_iff_forall_smul_nonneg.trans <| forall₄_congr fun i _ j _ ↦ by
    simp [smul_sub, sub_smul, ← add_sub_right_comm, le_sub_iff_add_le, add_comm (f i • g i),
      add_comm (f i • g j)]


/-- Two functions antivary iff the rearrangement inequality holds. -/
lemma antivaryOn_iff_smul_rearrangement :
    AntivaryOn f g s ↔
      ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → f i • g i + f j • g j ≤ f i • g j + f j • g i :=
                                           /-
                                             ι : Type u_1
                                             α : Type u_2
                                             β : Type u_3
                                             inst✝³ : LinearOrderedRing α
                                             inst✝² : LinearOrderedAddCommGroup β
                                             inst✝¹ : Module α β
                                             inst✝ : OrderedSMul α β
                                             f : ι → α
                                             g : ι → β
                                             s : Set ι
                                             ⊢ Iff (MonovaryOn f (Function.comp (⇑OrderDual.toDual) g) s) (∀ ⦃i : ι⦄, Membe …
                                           -/
  monovaryOn_toDual_right.symm.trans <| by rw [monovaryOn_iff_smul_rearrangement]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- Two functions monovary iff the rearrangement inequality holds. -/
lemma monovary_iff_smul_rearrangement :
    Monovary f g ↔ ∀ i j, f i • g j + f j • g i ≤ f i • g i + f j • g j :=
  monovaryOn_univ.symm.trans <| monovaryOn_iff_smul_rearrangement.trans <| by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : LinearOrderedRing α
      inst✝² : LinearOrderedAddCommGroup β
      inst✝¹ : Module α β
      inst✝ : OrderedSMul α β
      f : ι → α
      g : ι → β
      ⊢ Iff (∀ ⦃i : ι⦄, Membership.mem Set.univ i → ∀ ⦃j : ι⦄, Membership.mem Set.un …
    -/
    simp only [Set.mem_univ, forall_true_left]
    /-
      🎉 no goals
    -/


/-- Two functions antivary iff the rearrangement inequality holds. -/
lemma antivary_iff_smul_rearrangement :
    Antivary f g ↔ ∀ i j, f i • g i + f j • g j ≤ f i • g j + f j • g i :=
                                         /-
                                           ι : Type u_1
                                           α : Type u_2
                                           β : Type u_3
                                           inst✝³ : LinearOrderedRing α
                                           inst✝² : LinearOrderedAddCommGroup β
                                           inst✝¹ : Module α β
                                           inst✝ : OrderedSMul α β
                                           f : ι → α
                                           g : ι → β
                                           ⊢ Iff (Monovary f (Function.comp (⇑OrderDual.toDual) g)) (∀ (i j : ι), LE.le ( …
                                         -/
  monovary_toDual_right.symm.trans <| by rw [monovary_iff_smul_rearrangement]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


alias ⟨MonovaryOn.sub_smul_sub_nonneg, _⟩ := monovaryOn_iff_forall_smul_nonneg

alias ⟨AntivaryOn.sub_smul_sub_nonpos, _⟩ := antivaryOn_iff_forall_smul_nonpos

alias ⟨Monovary.sub_smul_sub_nonneg, _⟩ := monovary_iff_forall_smul_nonneg

alias ⟨Antivary.sub_smul_sub_nonpos, _⟩ := antivary_iff_forall_smul_nonpos

alias ⟨Monovary.smul_add_smul_le_smul_add_smul, _⟩ := monovary_iff_smul_rearrangement

alias ⟨Antivary.smul_add_smul_le_smul_add_smul, _⟩ := antivary_iff_smul_rearrangement

alias ⟨MonovaryOn.smul_add_smul_le_smul_add_smul, _⟩ := monovaryOn_iff_smul_rearrangement

alias ⟨AntivaryOn.smul_add_smul_le_smul_add_smul, _⟩ := antivaryOn_iff_smul_rearrangement


lemma monovaryOn_iff_forall_mul_nonneg :
    MonovaryOn f g s ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → 0 ≤ (f j - f i) * (g j - g i) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    s : Set ι
    ⊢ Iff (MonovaryOn f g s) (∀ ⦃i : ι⦄, Membership.mem s i → ∀ ⦃j : ι⦄, Membershi …
  -/
  simp only [smul_eq_mul, monovaryOn_iff_forall_smul_nonneg]
  /-
    🎉 no goals
  -/


lemma antivaryOn_iff_forall_mul_nonpos :
    AntivaryOn f g s ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → (f j - f i) * (g j - g i) ≤ 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    s : Set ι
    ⊢ Iff (AntivaryOn f g s) (∀ ⦃i : ι⦄, Membership.mem s i → ∀ ⦃j : ι⦄, Membershi …
  -/
  simp only [smul_eq_mul, antivaryOn_iff_forall_smul_nonpos]
  /-
    🎉 no goals
  -/


lemma monovary_iff_forall_mul_nonneg : Monovary f g ↔ ∀ i j, 0 ≤ (f j - f i) * (g j - g i) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    ⊢ Iff (Monovary f g) (∀ (i j : ι), LE.le 0 (HMul.hMul (HSub.hSub (f j) (f i))  …
  -/
  simp only [smul_eq_mul, monovary_iff_forall_smul_nonneg]
  /-
    🎉 no goals
  -/


lemma antivary_iff_forall_mul_nonpos : Antivary f g ↔ ∀ i j, (f j - f i) * (g j - g i) ≤ 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    ⊢ Iff (Antivary f g) (∀ (i j : ι), LE.le (HMul.hMul (HSub.hSub (f j) (f i)) (H …
  -/
  simp only [smul_eq_mul, antivary_iff_forall_smul_nonpos]
  /-
    🎉 no goals
  -/


/-- Two functions monovary iff the rearrangement inequality holds. -/
lemma monovaryOn_iff_mul_rearrangement :
    MonovaryOn f g s ↔
      ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → f i * g j + f j * g i ≤ f i * g i + f j * g j := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    s : Set ι
    ⊢ Iff (MonovaryOn f g s) (∀ ⦃i : ι⦄, Membership.mem s i → ∀ ⦃j : ι⦄, Membershi …
  -/
  simp only [smul_eq_mul, monovaryOn_iff_smul_rearrangement]
  /-
    🎉 no goals
  -/


/-- Two functions antivary iff the rearrangement inequality holds. -/
lemma antivaryOn_iff_mul_rearrangement :
    AntivaryOn f g s ↔
      ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ s → f i * g i + f j * g j ≤ f i * g j + f j * g i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    s : Set ι
    ⊢ Iff (AntivaryOn f g s) (∀ ⦃i : ι⦄, Membership.mem s i → ∀ ⦃j : ι⦄, Membershi …
  -/
  simp only [smul_eq_mul, antivaryOn_iff_smul_rearrangement]
  /-
    🎉 no goals
  -/


/-- Two functions monovary iff the rearrangement inequality holds. -/
lemma monovary_iff_mul_rearrangement :
    Monovary f g ↔ ∀ i j, f i * g j + f j * g i ≤ f i * g i + f j * g j := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    ⊢ Iff (Monovary f g) (∀ (i j : ι), LE.le (HAdd.hAdd (HMul.hMul (f i) (g j)) (H …
  -/
  simp only [smul_eq_mul, monovary_iff_smul_rearrangement]
  /-
    🎉 no goals
  -/


/-- Two functions antivary iff the rearrangement inequality holds. -/
lemma antivary_iff_mul_rearrangement :
    Antivary f g ↔ ∀ i j, f i * g i + f j * g j ≤ f i * g j + f j * g i := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedRing α
    f g : ι → α
    ⊢ Iff (Antivary f g) (∀ (i j : ι), LE.le (HAdd.hAdd (HMul.hMul (f i) (g i)) (H …
  -/
  simp only [smul_eq_mul, antivary_iff_smul_rearrangement]
  /-
    🎉 no goals
  -/


alias ⟨MonovaryOn.sub_mul_sub_nonneg, _⟩ := monovaryOn_iff_forall_mul_nonneg

alias ⟨AntivaryOn.sub_mul_sub_nonpos, _⟩ := antivaryOn_iff_forall_mul_nonpos

alias ⟨Monovary.sub_mul_sub_nonneg, _⟩ := monovary_iff_forall_mul_nonneg

alias ⟨Antivary.sub_mul_sub_nonpos, _⟩ := antivary_iff_forall_mul_nonpos

alias ⟨Monovary.mul_add_mul_le_mul_add_mul, _⟩ := monovary_iff_mul_rearrangement

alias ⟨Antivary.mul_add_mul_le_mul_add_mul, _⟩ := antivary_iff_mul_rearrangement

alias ⟨MonovaryOn.mul_add_mul_le_mul_add_mul, _⟩ := monovaryOn_iff_mul_rearrangement

alias ⟨AntivaryOn.mul_add_mul_le_mul_add_mul, _⟩ := antivaryOn_iff_mul_rearrangement


