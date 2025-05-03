/-- We say that `g : β → α` is an order right adjoint function for `f : α → β` if it sends each `y`
to a least upper bound for `{x | f x ≤ y}`. If `α` is a partial order, and `f : α → β` has
a right adjoint, then this right adjoint is unique. -/
def IsOrderRightAdjoint [Preorder α] [Preorder β] (f : α → β) (g : β → α) :=
  ∀ y, IsLUB { x | f x ≤ y } (g y)


theorem isOrderRightAdjoint_sSup [CompleteLattice α] [Preorder β] (f : α → β) :
    IsOrderRightAdjoint f fun y => sSup { x | f x ≤ y } := fun _ => isLUB_sSup _


theorem isOrderRightAdjoint_csSup [ConditionallyCompleteLattice α] [Preorder β] (f : α → β)
    (hne : ∀ y, ∃ x, f x ≤ y) (hbdd : ∀ y, BddAbove { x | f x ≤ y }) :
    IsOrderRightAdjoint f fun y => sSup { x | f x ≤ y } := fun y => isLUB_csSup (hne y) (hbdd y)


protected theorem unique [PartialOrder α] [Preorder β] {f : α → β} {g₁ g₂ : β → α}
    (h₁ : IsOrderRightAdjoint f g₁) (h₂ : IsOrderRightAdjoint f g₂) : g₁ = g₂ :=
  funext fun y => (h₁ y).unique (h₂ y)


theorem right_mono [Preorder α] [Preorder β] {f : α → β} {g : β → α} (h : IsOrderRightAdjoint f g) :
    Monotone g := fun y₁ y₂ hy => ((h y₁).mono (h y₂)) fun _ hx => le_trans hx hy


theorem orderIso_comp [Preorder α] [Preorder β] [Preorder γ] {f : α → β} {g : β → α}
    (h : IsOrderRightAdjoint f g) (e : β ≃o γ) : IsOrderRightAdjoint (e ∘ f) (g ∘ e.symm) :=
              /-
                α : Type u_1
                β : Type u_2
                γ : Type u_3
                inst✝² : Preorder α
                inst✝¹ : Preorder β
                inst✝ : Preorder γ
                f : α → β
                g : β → α
                h : IsOrderRightAdjoint f g
                e : OrderIso β γ
                y : γ
                ⊢ IsLUB (setOf fun x => LE.le (Function.comp (⇑e) f x) y) (Function.comp g (⇑e …
              -/
  fun y => by simpa [e.le_symm_apply] using h (e.symm y)
              /-
                🎉 no goals
              -/


theorem comp_orderIso [Preorder α] [Preorder β] [Preorder γ] {f : α → β} {g : β → α}
    (h : IsOrderRightAdjoint f g) (e : γ ≃o α) : IsOrderRightAdjoint (f ∘ e) (e.symm ∘ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → α
    h : IsOrderRightAdjoint f g
    e : OrderIso γ α
    ⊢ IsOrderRightAdjoint (Function.comp f ⇑e) (Function.comp (⇑e.symm) g)
  -/
  intro y
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → α
    h : IsOrderRightAdjoint f g
    e : OrderIso γ α
    y : β
    ⊢ IsLUB (setOf fun x => LE.le (Function.comp f (⇑e) x) y) (Function.comp (⇑e.s …
  -/
  change IsLUB (e ⁻¹' { x | f x ≤ y }) (e.symm (g y))
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → α
    h : IsOrderRightAdjoint f g
    e : OrderIso γ α
    y : β
    ⊢ IsLUB (Set.preimage (⇑e) (setOf fun x => LE.le (f x) y)) (e.symm (g y))
  -/
  rw [e.isLUB_preimage, e.apply_symm_apply]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → α
    h : IsOrderRightAdjoint f g
    e : OrderIso γ α
    y : β
    ⊢ IsLUB (setOf fun x => LE.le (f x) y) (g y)
  -/
  exact h y
  /-
    🎉 no goals
  -/


/-- If an order automorphism `fa` is semiconjugate to an order embedding `fb` by a function `g`
and `g'` is an order right adjoint of `g` (i.e. `g' y = sSup {x | f x ≤ y}`), then `fb` is
semiconjugate to `fa` by `g'`.

This is a version of Proposition 2.1 from [Étienne Ghys, Groupes d'homéomorphismes du cercle et
cohomologie bornée][ghys87:groupes]. -/
theorem Semiconj.symm_adjoint [PartialOrder α] [Preorder β] {fa : α ≃o α} {fb : β ↪o β} {g : α → β}
    (h : Function.Semiconj g fa fb) {g' : β → α} (hg' : IsOrderRightAdjoint g g') :
    Function.Semiconj g' fb fa := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    fa : OrderIso α α
    fb : OrderEmbedding β β
    g : α → β
    h : Function.Semiconj g ⇑fa ⇑fb
    g' : β → α
    hg' : IsOrderRightAdjoint g g'
    ⊢ Function.Semiconj g' ⇑fb ⇑fa
  -/
  refine fun y => (hg' _).unique ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    fa : OrderIso α α
    fb : OrderEmbedding β β
    g : α → β
    h : Function.Semiconj g ⇑fa ⇑fb
    g' : β → α
    hg' : IsOrderRightAdjoint g g'
    y : β
    ⊢ IsLUB (setOf fun x => LE.le (g x) (fb y)) (fa (g' y))
  -/
  rw [← fa.surjective.image_preimage { x | g x ≤ fb y }, preimage_setOf_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : Preorder β
    fa : OrderIso α α
    fb : OrderEmbedding β β
    g : α → β
    h : Function.Semiconj g ⇑fa ⇑fb
    g' : β → α
    hg' : IsOrderRightAdjoint g g'
    y : β
    ⊢ IsLUB (Set.image (⇑fa) (setOf fun a => LE.le (g (fa a)) (fb y))) (fa (g' y))
  -/
  simp only [h.eq, fb.le_iff_le, fa.leftOrdContinuous (hg' _)]
  /-
    🎉 no goals
  -/


theorem semiconj_of_isLUB [PartialOrder α] [Group G] (f₁ f₂ : G →* α ≃o α) {h : α → α}
    (H : ∀ x, IsLUB (range fun g' => (f₁ g')⁻¹ (f₂ g' x)) (h x)) (g : G) :
    Function.Semiconj h (f₂ g) (f₁ g) := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : Group G
    f₁ f₂ : MonoidHom G (OrderIso α α)
    h : α → α
    H : ∀ (x : α), IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') x)) (h x)
    g : G
    ⊢ Function.Semiconj h ⇑(f₂ g) ⇑(f₁ g)
  -/
  refine fun y => (H _).unique ?_
  /-
    α : Type u_1
    G : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : Group G
    f₁ f₂ : MonoidHom G (OrderIso α α)
    h : α → α
    H : ∀ (x : α), IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') x)) (h x)
    g : G
    y : α
    ⊢ IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') ((f₂ g) y))) ((f₁ g) ( …
  -/
  have := (f₁ g).leftOrdContinuous (H y)
  /-
    α : Type u_1
    G : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : Group G
    f₁ f₂ : MonoidHom G (OrderIso α α)
    h : α → α
    H : ∀ (x : α), IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') x)) (h x)
    g : G
    y : α
    this : IsLUB (Set.image (⇑(f₁ g)) (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂  …
    ⊢ IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') ((f₂ g) y))) ((f₁ g) ( …
  -/
  rw [← range_comp, ← (Equiv.mulRight g).surjective.range_comp _] at this
  /-
    α : Type u_1
    G : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : Group G
    f₁ f₂ : MonoidHom G (OrderIso α α)
    h : α → α
    H : ∀ (x : α), IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') x)) (h x)
    g : G
    y : α
    this : IsLUB (Set.range (Function.comp (Function.comp ⇑(f₁ g) fun g' => (Inv.i …
    ⊢ IsLUB (Set.range fun g' => (Inv.inv (f₁ g')) ((f₂ g') ((f₂ g) y))) ((f₁ g) ( …
  -/
  simpa [comp_def] using this
  /-
    🎉 no goals
  -/


/-- Consider two actions `f₁ f₂ : G → α → α` of a group on a complete lattice by order
isomorphisms. Then the map `x ↦ ⨆ g : G, (f₁ g)⁻¹ (f₂ g x)` semiconjugates each `f₁ g'` to `f₂ g'`.

This is a version of Proposition 5.4 from [Étienne Ghys, Groupes d'homéomorphismes du cercle et
cohomologie bornée][ghys87:groupes]. -/
theorem sSup_div_semiconj [CompleteLattice α] [Group G] (f₁ f₂ : G →* α ≃o α) (g : G) :
    Function.Semiconj (fun x => ⨆ g' : G, (f₁ g')⁻¹ (f₂ g' x)) (f₂ g) (f₁ g) :=
  semiconj_of_isLUB f₁ f₂ (fun _ => isLUB_iSup) _


/-- Consider two actions `f₁ f₂ : G → α → α` of a group on a conditionally complete lattice by order
isomorphisms. Suppose that each set $s(x)=\{f_1(g)^{-1} (f_2(g)(x)) | g \in G\}$ is bounded above.
Then the map `x ↦ sSup s(x)` semiconjugates each `f₁ g'` to `f₂ g'`.

This is a version of Proposition 5.4 from [Étienne Ghys, Groupes d'homéomorphismes du cercle et
cohomologie bornée][ghys87:groupes]. -/
theorem csSup_div_semiconj [ConditionallyCompleteLattice α] [Group G] (f₁ f₂ : G →* α ≃o α)
    (hbdd : ∀ x, BddAbove (range fun g => (f₁ g)⁻¹ (f₂ g x))) (g : G) :
    Function.Semiconj (fun x => ⨆ g' : G, (f₁ g')⁻¹ (f₂ g' x)) (f₂ g) (f₁ g) :=
  semiconj_of_isLUB f₁ f₂ (fun x => isLUB_csSup (range_nonempty _) (hbdd x)) _


