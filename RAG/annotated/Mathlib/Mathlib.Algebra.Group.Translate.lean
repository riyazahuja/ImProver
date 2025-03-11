/-- Translation of a function in a group by an element of that group.
`τ a f` is defined as `x ↦ f (x - a)`.  -/
def translate (a : G) (f : G → α) : G → α := fun x ↦ f (x - a)


@[inherit_doc] scoped[translate] notation "τ " => translate


@[simp] lemma translate_apply (a : G) (f : G → α) (x : G) : τ a f x = f (x - a) := rfl

                                                           /-
                                                             α : Type u_2
                                                             G : Type u_5
                                                             inst✝ : AddCommGroup G
                                                             f : G → α
                                                             ⊢ Eq (translate 0 f) f
                                                           -/
@[simp] lemma translate_zero (f : G → α) : τ 0 f = f := by ext; simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma translate_translate (a b : G) (f : G → α) : τ a (τ b f) = τ (a + b) f := by
  /-
    α : Type u_2
    G : Type u_5
    inst✝ : AddCommGroup G
    a b : G
    f : G → α
    ⊢ Eq (translate a (translate b f)) (translate (HAdd.hAdd a b) f)
  -/
  ext; simp [sub_sub]
       /-
         🎉 no goals
       -/


                                                                            /-
                                                                              α : Type u_2
                                                                              G : Type u_5
                                                                              inst✝ : AddCommGroup G
                                                                              a b : G
                                                                              f : G → α
                                                                              ⊢ Eq (translate (HAdd.hAdd a b) f) (translate a (translate b f))
                                                                            -/
lemma translate_add (a b : G) (f : G → α) : τ (a + b) f = τ a (τ b f) := by ext; simp [sub_sub]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- See `translate_add`-/
lemma translate_add' (a b : G) (f : G → α) : τ (a + b) f = τ b (τ a f) := by
  /-
    α : Type u_2
    G : Type u_5
    inst✝ : AddCommGroup G
    a b : G
    f : G → α
    ⊢ Eq (translate (HAdd.hAdd a b) f) (translate b (translate a f))
  -/
  rw [add_comm, translate_add]
  /-
    🎉 no goals
  -/


lemma translate_comm (a b : G) (f : G → α) : τ a (τ b f) = τ b (τ a f) := by
  /-
    α : Type u_2
    G : Type u_5
    inst✝ : AddCommGroup G
    a b : G
    f : G → α
    ⊢ Eq (translate a (translate b f)) (translate b (translate a f))
  -/
  rw [← translate_add, translate_add']
  /-
    🎉 no goals
  -/

-- We make `simp` push the `τ` outside

@[simp] lemma comp_translate (a : G) (f : G → α) (g : α → β) : g ∘ τ a f = τ a (g ∘ f) := rfl


lemma translate_eq_domAddActMk_vadd (a : G) (f : G → α) : τ a f = DomAddAct.mk (-a) +ᵥ f := by
  /-
    α : Type u_2
    G : Type u_5
    inst✝ : AddCommGroup G
    a : G
    f : G → α
    ⊢ Eq (translate a f) (HVAdd.hVAdd (DomAddAct.mk (Neg.neg a)) f)
  -/
  ext; simp [DomAddAct.vadd_apply, sub_eq_neg_add]
       /-
         🎉 no goals
       -/


@[simp]
lemma translate_smul_right [SMul H α] (a : G) (f : G → α) (c : H) : τ a (c • f) = c • τ a f := rfl


@[simp] lemma translate_zero_right [Zero α] (a : G) : τ a (0 : G → α) = 0 := rfl

lemma translate_add_right [Add α] (a : G) (f g : G → α) : τ a (f + g) = τ a f + τ a g := rfl

lemma translate_sub_right [Sub α] (a : G) (f g : G → α) : τ a (f - g) = τ a f - τ a g := rfl

lemma translate_neg_right [Neg α] (a : G) (f : G → α) : τ a (-f) = -τ a f := rfl


lemma translate_sum_right (a : G) (f : ι → G → M) (s : Finset ι) :
                                                    /-
                                                      ι : Type u_1
                                                      M : Type u_4
                                                      G : Type u_5
                                                      inst✝¹ : AddCommGroup G
                                                      inst✝ : AddCommMonoid M
                                                      a : G
                                                      f : ι → G → M
                                                      s : Finset ι
                                                      ⊢ Eq (translate a (s.sum fun i => f i)) (s.sum fun i => translate a (f i))
                                                    -/
    τ a (∑ i in s, f i) = ∑ i in s, τ a (f i) := by ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma sum_translate [Fintype G] (a : G) (f : G → M) : ∑ b, τ a f b = ∑ b, f b :=
  Fintype.sum_equiv (Equiv.subRight _) _ _ fun _ ↦ rfl


@[simp] lemma support_translate (a : G) (f : G → H) : support (τ a f) = a +ᵥ support f := by
  /-
    G : Type u_5
    H : Type u_6
    inst✝¹ : AddCommGroup G
    inst✝ : AddCommGroup H
    a : G
    f : G → H
    ⊢ Eq (Function.support (translate a f)) (HVAdd.hVAdd a (Function.support f))
  -/
  ext; simp [mem_vadd_set_iff_neg_vadd_mem, sub_eq_neg_add]
       /-
         🎉 no goals
       -/


lemma translate_prod_right (a : G) (f : ι → G → M) (s : Finset ι) :
                                                    /-
                                                      ι : Type u_1
                                                      M : Type u_4
                                                      G : Type u_5
                                                      inst✝¹ : AddCommGroup G
                                                      inst✝ : CommMonoid M
                                                      a : G
                                                      f : ι → G → M
                                                      s : Finset ι
                                                      ⊢ Eq (translate a (s.prod fun i => f i)) (s.prod fun i => translate a (f i))
                                                    -/
    τ a (∏ i in s, f i) = ∏ i in s, τ a (f i) := by ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/

