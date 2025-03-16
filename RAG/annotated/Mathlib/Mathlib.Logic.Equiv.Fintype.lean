/-- Computably turn an embedding `f : α ↪ β` into an equiv `α ≃ Set.range f`,
if `α` is a `Fintype`. Has poor computational performance, due to exhaustive searching in
constructed inverse. When a better inverse is known, use `Equiv.ofLeftInverse'` or
`Equiv.ofLeftInverse` instead. This is the computable version of `Equiv.ofInjective`.
-/
def Function.Embedding.toEquivRange : α ≃ Set.range f :=
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        inst✝¹ : Fintype α
                                                                        inst✝ : DecidableEq β
                                                                        e : Equiv.Perm α
                                                                        f : Function.Embedding α β
                                                                        x✝ : α
                                                                        ⊢ Eq (f.invOfMemRange ((fun a => ⟨f a, ⋯⟩) x✝)) x✝
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  ⟨fun a => ⟨f a, Set.mem_range_self a⟩, f.invOfMemRange, fun _ => by simp, fun _ => by simp⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[simp]
theorem Function.Embedding.toEquivRange_apply (a : α) :
    f.toEquivRange a = ⟨f a, Set.mem_range_self a⟩ :=
  rfl


@[simp]
theorem Function.Embedding.toEquivRange_symm_apply_self (a : α) :
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝¹ : Fintype α
                                                                inst✝ : DecidableEq β
                                                                f : Function.Embedding α β
                                                                a : α
                                                                ⊢ Eq (f.toEquivRange.symm ⟨f a, ⋯⟩) a
                                                              -/
    f.toEquivRange.symm ⟨f a, Set.mem_range_self a⟩ = a := by simp [Equiv.symm_apply_eq]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem Function.Embedding.toEquivRange_eq_ofInjective :
    f.toEquivRange = Equiv.ofInjective f f.injective := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    f : Function.Embedding α β
    ⊢ Eq f.toEquivRange (Equiv.ofInjective ⇑f ⋯)
  -/
  ext
  /-
    case H.a
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    f : Function.Embedding α β
    x✝ : α
    ⊢ Eq ↑(f.toEquivRange x✝) ↑((Equiv.ofInjective ⇑f ⋯) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Extend the domain of `e : Equiv.Perm α`, mapping it through `f : α ↪ β`.
Everything outside of `Set.range f` is kept fixed. Has poor computational performance,
due to exhaustive searching in constructed inverse due to using `Function.Embedding.toEquivRange`.
When a better `α ≃ Set.range f` is known, use `Equiv.Perm.viaSetRange`.
When `[Fintype α]` is not available, a noncomputable version is available as
`Equiv.Perm.viaEmbedding`.
-/
def Equiv.Perm.viaFintypeEmbedding : Equiv.Perm β :=
  e.extendDomain f.toEquivRange


@[simp]
theorem Equiv.Perm.viaFintypeEmbedding_apply_image (a : α) :
    e.viaFintypeEmbedding f (f a) = f (e a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    a : α
    ⊢ Eq ((e.viaFintypeEmbedding f) (f a)) (f (e a))
  -/
  rw [Equiv.Perm.viaFintypeEmbedding]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    a : α
    ⊢ Eq ((e.extendDomain f.toEquivRange) (f a)) (f (e a))
  -/
  convert Equiv.Perm.extendDomain_apply_image e (Function.Embedding.toEquivRange f) a
  /-
    🎉 no goals
  -/


theorem Equiv.Perm.viaFintypeEmbedding_apply_mem_range {b : β} (h : b ∈ Set.range f) :
    e.viaFintypeEmbedding f b = f (e (f.invOfMemRange ⟨b, h⟩)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    b : β
    h : Membership.mem (Set.range ⇑f) b
    ⊢ Eq ((e.viaFintypeEmbedding f) b) (f (e (f.invOfMemRange ⟨b, h⟩)))
  -/
  simp only [viaFintypeEmbedding, Function.Embedding.invOfMemRange]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    b : β
    h : Membership.mem (Set.range ⇑f) b
    ⊢ Eq ((e.extendDomain f.toEquivRange) b) (f (e (⋯.invOfMemRange ⟨b, h⟩)))
  -/
  rw [Equiv.Perm.extendDomain_apply_subtype]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    b : β
    h : Membership.mem (Set.range ⇑f) b
    ⊢ Eq (↑(f.toEquivRange (e (f.toEquivRange.symm ⟨b, ?h⟩)))) (f (e (⋯.invOfMemRa …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem Equiv.Perm.viaFintypeEmbedding_apply_not_mem_range {b : β} (h : b ∉ Set.range f) :
    e.viaFintypeEmbedding f b = b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    b : β
    h : Not (Membership.mem (Set.range ⇑f) b)
    ⊢ Eq ((e.viaFintypeEmbedding f) b) b
  -/
  rwa [Equiv.Perm.viaFintypeEmbedding, Equiv.Perm.extendDomain_apply_not_subtype]
  /-
    🎉 no goals
  -/


@[simp]
theorem Equiv.Perm.viaFintypeEmbedding_sign [DecidableEq α] [Fintype β] :
    Equiv.Perm.sign (e.viaFintypeEmbedding f) = Equiv.Perm.sign e := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : DecidableEq β
    e : Equiv.Perm α
    f : Function.Embedding α β
    inst✝¹ : DecidableEq α
    inst✝ : Fintype β
    ⊢ Eq (Equiv.Perm.sign (e.viaFintypeEmbedding f)) (Equiv.Perm.sign e)
  -/
  simp [Equiv.Perm.viaFintypeEmbedding]
  /-
    🎉 no goals
  -/


/-- If `e` is an equivalence between two subtypes of a finite type `α`, `e.toCompl`
is an equivalence between the complement of those subtypes.

See also `Equiv.compl`, for a computable version when a term of type
`{e' : α ≃ α // ∀ x : {x // p x}, e' x = e x}` is known. -/
noncomputable def toCompl {p q : α → Prop} (e : { x // p x } ≃ { x // q x }) :
    { x // ¬p x } ≃ { x // ¬q x } := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite α
    p q : α → Prop
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    ⊢ Equiv (Subtype fun x => Not (p x)) (Subtype fun x => Not (q x))
  -/
  apply Classical.choice
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝ : Finite α
    p q : α → Prop
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    ⊢ Nonempty (Equiv (Subtype fun x => Not (p x)) (Subtype fun x => Not (q x)))
  -/
  cases nonempty_fintype α
  classical
  exact Fintype.card_eq.mp <| Fintype.card_compl_eq_card_compl _ _ <| Fintype.card_congr e


/-- If `e` is an equivalence between two subtypes of a fintype `α`, `e.extendSubtype`
is a permutation of `α` acting like `e` on the subtypes and doing something arbitrary outside.

Note that when `p = q`, `Equiv.Perm.subtypeCongr e (Equiv.refl _)` can be used instead. -/
noncomputable abbrev extendSubtype (e : { x // p x } ≃ { x // q x }) : Perm α :=
  subtypeCongr e e.toCompl


theorem extendSubtype_apply_of_mem (e : { x // p x } ≃ { x // q x }) (x) (hx : p x) :
    e.extendSubtype x = e ⟨x, hx⟩ := by
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : p x
    ⊢ Eq (e.extendSubtype x) ↑(e ⟨x, hx⟩)
  -/
  dsimp only [extendSubtype]
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : p x
    ⊢ Eq ((e.subtypeCongr e.toCompl) x) ↑(e ⟨x, hx⟩)
  -/
  simp only [subtypeCongr, Equiv.trans_apply, Equiv.sumCongr_apply]
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : p x
    ⊢ Eq ((Equiv.sumCompl q) (Sum.map (⇑e) (⇑e.toCompl) ((Equiv.sumCompl p).symm x …
  -/
  rw [sumCompl_apply_symm_of_pos _ _ hx, Sum.map_inl, sumCompl_apply_inl]
  /-
    🎉 no goals
  -/


theorem extendSubtype_mem (e : { x // p x } ≃ { x // q x }) (x) (hx : p x) :
    q (e.extendSubtype x) := by
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : p x
    ⊢ q (e.extendSubtype x)
  -/
  convert (e ⟨x, hx⟩).2
  /-
    case h.e'_1
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : p x
    ⊢ Eq (e.extendSubtype x) ↑(e ⟨x, hx⟩)
  -/
  rw [e.extendSubtype_apply_of_mem _ hx]
  /-
    🎉 no goals
  -/


theorem extendSubtype_apply_of_not_mem (e : { x // p x } ≃ { x // q x }) (x) (hx : ¬p x) :
    e.extendSubtype x = e.toCompl ⟨x, hx⟩ := by
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : Not (p x)
    ⊢ Eq (e.extendSubtype x) ↑(e.toCompl ⟨x, hx⟩)
  -/
  dsimp only [extendSubtype]
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : Not (p x)
    ⊢ Eq ((e.subtypeCongr e.toCompl) x) ↑(e.toCompl ⟨x, hx⟩)
  -/
  simp only [subtypeCongr, Equiv.trans_apply, Equiv.sumCongr_apply]
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : Not (p x)
    ⊢ Eq ((Equiv.sumCompl q) (Sum.map (⇑e) (⇑e.toCompl) ((Equiv.sumCompl p).symm x …
  -/
  rw [sumCompl_apply_symm_of_neg _ _ hx, Sum.map_inr, sumCompl_apply_inr]
  /-
    🎉 no goals
  -/


theorem extendSubtype_not_mem (e : { x // p x } ≃ { x // q x }) (x) (hx : ¬p x) :
    ¬q (e.extendSubtype x) := by
  /-
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : Not (p x)
    ⊢ Not (q (e.extendSubtype x))
  -/
  convert (e.toCompl ⟨x, hx⟩).2
  /-
    case h.e'_1.h.e'_1
    α : Type u_1
    inst✝² : Finite α
    p q : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    e : Equiv (Subtype fun x => p x) (Subtype fun x => q x)
    x : α
    hx : Not (p x)
    ⊢ Eq (e.extendSubtype x) ↑(e.toCompl ⟨x, hx⟩)
  -/
  rw [e.extendSubtype_apply_of_not_mem _ hx]
  /-
    🎉 no goals
  -/


