lemma mem_stabilizer_iff {g : (Perm α)ᵈᵐᵃ} :
    g ∈ stabilizer (Perm α)ᵈᵐᵃ f ↔ f ∘ (mk.symm g :) = f := by
  /-
    α : Type u_1
    ι : Type u_2
    f : α → ι
    g : DomMulAct (Equiv.Perm α)
    ⊢ Iff (Membership.mem (MulAction.stabilizer (DomMulAct (Equiv.Perm α)) f) g) ( …
  -/
  simp only [MulAction.mem_stabilizer_iff]; rfl
                                            /-
                                              🎉 no goals
                                            -/


/-- The `invFun` component of `MulEquiv` from `MulAction.stabilizer (Perm α) f`
  to the product of the `Equiv.Perm {a // f a = i} -/
def stabilizerEquiv_invFun (g : ∀ i, Perm {a // f a = i}) (a : α) : α := g (f a) ⟨a, rfl⟩


lemma stabilizerEquiv_invFun_eq (g : ∀ i, Perm {a // f a = i}) {a : α} {i : ι} (h : f a = i) :
                                                  /-
                                                    α : Type u_1
                                                    ι : Type u_2
                                                    f : α → ι
                                                    g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
                                                    a : α
                                                    i : ι
                                                    h : Eq (f a) i
                                                    ⊢ Eq (DomMulAct.stabilizerEquiv_invFun g a) ↑((g i) ⟨a, h⟩)
                                                  -/
    stabilizerEquiv_invFun g a = g i ⟨a, h⟩ := by subst h; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma comp_stabilizerEquiv_invFun (g : ∀ i, Perm {a // f a = i}) (a : α) :
    f (stabilizerEquiv_invFun g a) = f a :=
  (g (f a) ⟨a, rfl⟩).prop


/-- The `invFun` component of `MulEquiv` from `MulAction.stabilizer (Perm α) p`
  to the product of the `Equiv.Perm {a | f a = i} (as an `Equiv.Perm α`) -/
def stabilizerEquiv_invFun_aux (g : ∀ i, Perm {a // f a = i}) : Perm α where
  toFun := stabilizerEquiv_invFun g
  invFun := stabilizerEquiv_invFun (fun i ↦ (g i).symm)
  left_inv a := by
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (DomMulAct.stabilizerEquiv_invFun (fun i => Equiv.symm (g i)) (DomMulAct. …
    -/
    rw [stabilizerEquiv_invFun_eq _ (comp_stabilizerEquiv_invFun g a)]
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (↑((Equiv.symm (g (f a))) ⟨DomMulAct.stabilizerEquiv_invFun g a, ⋯⟩)) a
    -/
    exact congr_arg Subtype.val ((g <| f a).left_inv _)
    /-
      🎉 no goals
    -/
  right_inv a := by
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (DomMulAct.stabilizerEquiv_invFun g (DomMulAct.stabilizerEquiv_invFun (fu …
    -/
    rw [stabilizerEquiv_invFun_eq _ (comp_stabilizerEquiv_invFun _ a)]
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (↑((g (f a)) ⟨DomMulAct.stabilizerEquiv_invFun (fun i => Equiv.symm (g i) …
    -/
    exact congr_arg Subtype.val ((g <| f a).right_inv _)
    /-
      🎉 no goals
    -/


/-- The `MulEquiv` from the `MulOpposite` of `MulAction.stabilizer (Perm α)ᵈᵐᵃ f`
  to the product of the `Equiv.Perm {a // f a = i}` -/
def stabilizerMulEquiv : (stabilizer (Perm α)ᵈᵐᵃ f)ᵐᵒᵖ ≃* (∀ i, Perm {a // f a = i}) where
  toFun g i := Perm.subtypePerm (mk.symm g.unop) fun a ↦ by
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : MulOpposite (Subtype fun x => Membership.mem (MulAction.stabilizer (DomMul …
      i : ι
      a : α
      ⊢ Iff (Eq (f a) i) (Eq (f ((DomMulAct.mk.symm ↑(MulOpposite.unop g)) a)) i)
    -/
    rw [← Function.comp_apply (f := f), mem_stabilizer_iff.mp g.unop.prop]
    /-
      🎉 no goals
    -/
  invFun g := ⟨mk (stabilizerEquiv_invFun_aux g), by
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      ⊢ Membership.mem (MulAction.stabilizer (DomMulAct (Equiv.Perm α)) f) (DomMulAc …
    -/
    ext a
    /-
      case h
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (HSMul.hSMul (DomMulAct.mk (DomMulAct.stabilizerEquiv_invFun_aux g)) f a) …
    -/
    rw [smul_apply, symm_apply_apply, Perm.smul_def]
    /-
      case h
      α : Type u_1
      ι : Type u_2
      f : α → ι
      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
      a : α
      ⊢ Eq (f ((DomMulAct.stabilizerEquiv_invFun_aux g) a)) (f a)
    -/
    apply comp_stabilizerEquiv_invFun⟩
    /-
      🎉 no goals
    -/
  left_inv _ := rfl
                    /-
                      α : Type u_1
                      ι : Type u_2
                      f : α → ι
                      g : (i : ι) → Equiv.Perm (Subtype fun a => Eq (f a) i)
                      ⊢ Eq ((fun g i => (DomMulAct.mk.symm ↑(MulOpposite.unop g)).subtypePerm ⋯) ((f …
                    -/
  right_inv g := by ext i a; apply stabilizerEquiv_invFun_eq
                             /-
                               🎉 no goals
                             -/
  map_mul' _ _ := rfl


lemma stabilizerMulEquiv_apply (g : (stabilizer (Perm α)ᵈᵐᵃ f)ᵐᵒᵖ) {a : α} {i : ι} (h : f a = i) :
    ((stabilizerMulEquiv f)) g i ⟨a, h⟩ = (mk.symm g.unop : Equiv.Perm α) a := rfl


/-- The cardinality of the type of permutations preserving a function -/
theorem stabilizer_card [DecidableEq α] [DecidableEq ι] [Fintype ι] :
    Fintype.card {g : Perm α // f ∘ g = f} = ∏ i, (Fintype.card {a // f a = i})! := by
  -- rewriting via Nat.card because Fintype instance is not found
  rw [← Nat.card_eq_fintype_card,
    Nat.card_congr (subtypeEquiv mk fun _ ↦ ?_),
    Nat.card_congr MulOpposite.opEquiv,
    Nat.card_congr (DomMulAct.stabilizerMulEquiv f).toEquiv, Nat.card_pi]
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      ⊢ Eq (Finset.univ.prod fun a => Nat.card (Equiv.Perm (Subtype fun a_1 => Eq (f …
    -/
  · exact Finset.prod_congr rfl fun i _ ↦ by rw [Nat.card_eq_fintype_card, Fintype.card_perm]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      x✝ : Equiv.Perm α
      ⊢ Iff (Eq (Function.comp f ⇑x✝) f) (Membership.mem (MulAction.stabilizer (DomM …
    -/
  · rfl
    /-
      🎉 no goals
    -/


omit [Fintype α] in
/-- The cardinality of the set of permutations preserving a function -/
theorem stabilizer_ncard [Finite α] [Fintype ι] :
    Set.ncard {g : Perm α | f ∘ g = f} = ∏ i, (Set.ncard {a | f a = i})! := by
  classical
  cases nonempty_fintype α
  simp only [← Set.Nat.card_coe_set_eq, Set.coe_setOf, card_eq_fintype_card]
  exact stabilizer_card f


/-- The cardinality of the type of permutations preserving a function
  (without the finiteness assumption on target)-/
theorem stabilizer_card':
    Fintype.card {g : Perm α // f ∘ g = f} =
      ∏ i in Finset.univ.image f, (Fintype.card ({a // f a = i}))! := by
  set φ : α → Finset.univ.image f :=
    Set.codRestrict f (Finset.univ.image f) (fun a => by simp)
  suffices ∀ g : Perm α, f ∘ g = f ↔ φ ∘ g = φ by
    simp only [this, stabilizer_card]
    apply Finset.prod_bij (fun g _ => g.val)
    · exact fun g _ => Finset.coe_mem g
    · exact fun g _ g' _ =>  SetCoe.ext
    · exact fun g hg => by
        rw [Finset.mem_image] at hg
        obtain ⟨a, _, rfl⟩ := hg
        use ⟨f a, by simp only [Finset.mem_image, Finset.mem_univ, true_and, exists_apply_eq_apply]⟩
        simp only [Finset.univ_eq_attach, Finset.mem_attach, exists_const]
    · intro i _
      apply congr_arg
      apply Fintype.card_congr
      apply Equiv.subtypeEquiv (Equiv.refl α)
      intro a
      rw [refl_apply, ← Subtype.coe_inj]
      simp only [φ, Set.val_codRestrict_apply]
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq ι
      φ : α → Subtype fun x => Membership.mem (Finset.image f Finset.univ) x := Set. …
      ⊢ ∀ (g : Equiv.Perm α), Iff (Eq (Function.comp f ⇑g) f) (Eq (Function.comp φ ⇑ …
    -/
  · intro g
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq ι
      φ : α → Subtype fun x => Membership.mem (Finset.image f Finset.univ) x := Set. …
      g : Equiv.Perm α
      ⊢ Iff (Eq (Function.comp f ⇑g) f) (Eq (Function.comp φ ⇑g) φ)
    -/
    simp only [funext_iff]
    /-
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq ι
      φ : α → Subtype fun x => Membership.mem (Finset.image f Finset.univ) x := Set. …
      g : Equiv.Perm α
      ⊢ Iff (∀ (x : α), Eq (Function.comp f (⇑g) x) (f x)) (∀ (x : α), Eq (Function. …
    -/
    apply forall_congr'
    /-
      case h
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq ι
      φ : α → Subtype fun x => Membership.mem (Finset.image f Finset.univ) x := Set. …
      g : Equiv.Perm α
      ⊢ ∀ (a : α), Iff (Eq (Function.comp f (⇑g) a) (f a)) (Eq (Function.comp φ (⇑g) …
    -/
    intro a
    /-
      case h
      α : Type u_1
      ι : Type u_2
      f : α → ι
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq ι
      φ : α → Subtype fun x => Membership.mem (Finset.image f Finset.univ) x := Set. …
      g : Equiv.Perm α
      a : α
      ⊢ Iff (Eq (Function.comp f (⇑g) a) (f a)) (Eq (Function.comp φ (⇑g) a) (φ a))
    -/
    simp only [Function.comp_apply, φ, ← Subtype.coe_inj, Set.val_codRestrict_apply]
    /-
      🎉 no goals
    -/


