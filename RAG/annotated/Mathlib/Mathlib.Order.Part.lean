lemma Monotone.partBind (hf : Monotone f) (hg : Monotone g) :
    Monotone fun x ↦ (f x).bind (g x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Monotone f
    hg : Monotone g
    ⊢ Monotone fun x => (f x).bind (g x)
  -/
  rintro x y h a
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Monotone f
    hg : Monotone g
    x y : α
    h : LE.le x y
    a : γ
    ⊢ Membership.mem ((fun x => (f x).bind (g x)) x) a → Membership.mem ((fun x => …
  -/
  simp only [and_imp, exists_prop, Part.bind_eq_bind, Part.mem_bind_iff, exists_imp]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Monotone f
    hg : Monotone g
    x y : α
    h : LE.le x y
    a : γ
    ⊢ ∀ (x_1 : β), Membership.mem (f x) x_1 → Membership.mem (g x x_1) a → Exists  …
  -/
  exact fun b hb ha ↦ ⟨b, hf h _ hb, hg h _ _ ha⟩
  /-
    🎉 no goals
  -/


lemma Antitone.partBind (hf : Antitone f) (hg : Antitone g) :
    Antitone fun x ↦ (f x).bind (g x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Antitone f
    hg : Antitone g
    ⊢ Antitone fun x => (f x).bind (g x)
  -/
  rintro x y h a
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Antitone f
    hg : Antitone g
    x y : α
    h : LE.le x y
    a : γ
    ⊢ Membership.mem ((fun x => (f x).bind (g x)) y) a → Membership.mem ((fun x => …
  -/
  simp only [and_imp, exists_prop, Part.bind_eq_bind, Part.mem_bind_iff, exists_imp]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : α → Part β
    g : α → β → Part γ
    hf : Antitone f
    hg : Antitone g
    x y : α
    h : LE.le x y
    a : γ
    ⊢ ∀ (x_1 : β), Membership.mem (f y) x_1 → Membership.mem (g y x_1) a → Exists  …
  -/
  exact fun b hb ha ↦ ⟨b, hf h _ hb, hg h _ _ ha⟩
  /-
    🎉 no goals
  -/


lemma Monotone.partMap (hg : Monotone g) : Monotone fun x ↦ (g x).map f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : β → γ
    g : α → Part β
    hg : Monotone g
    ⊢ Monotone fun x => Part.map f (g x)
  -/
  simpa only [← bind_some_eq_map] using hg.partBind monotone_const
  /-
    🎉 no goals
  -/


lemma Antitone.partMap (hg : Antitone g) : Antitone fun x ↦ (g x).map f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    f : β → γ
    g : α → Part β
    hg : Antitone g
    ⊢ Antitone fun x => Part.map f (g x)
  -/
  simpa only [← bind_some_eq_map] using hg.partBind antitone_const
  /-
    🎉 no goals
  -/


lemma Monotone.partSeq (hf : Monotone f) (hg : Monotone g) : Monotone fun x ↦ f x <*> g x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    β γ : Type u_4
    f : α → Part (β → γ)
    g : α → Part β
    hf : Monotone f
    hg : Monotone g
    ⊢ Monotone fun x => Seq.seq (f x) fun x_1 => g x
  -/
  simpa only [seq_eq_bind_map] using hf.partBind <| Monotone.of_apply₂ fun _ ↦ hg.partMap
  /-
    🎉 no goals
  -/


lemma Antitone.partSeq (hf : Antitone f) (hg : Antitone g) : Antitone fun x ↦ f x <*> g x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    β γ : Type u_4
    f : α → Part (β → γ)
    g : α → Part β
    hf : Antitone f
    hg : Antitone g
    ⊢ Antitone fun x => Seq.seq (f x) fun x_1 => g x
  -/
  simpa only [seq_eq_bind_map] using hf.partBind <| Antitone.of_apply₂ fun _ ↦ hg.partMap
  /-
    🎉 no goals
  -/


/-- `Part.bind` as a monotone function -/
@[simps]
def partBind (f : α →o Part β) (g : α →o β → Part γ) : α →o Part γ where
  toFun x := (f x).bind (g x)
  monotone' := f.2.partBind g.2


@[deprecated (since := "2024-07-04")] alias bind := partBind


