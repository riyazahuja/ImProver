/-- A function `f` is monotone if `a ≤ b` implies `f a ≤ f b`. -/
def Monotone (f : α → β) : Prop :=
  ∀ ⦃a b⦄, a ≤ b → f a ≤ f b


/-- A function `f` is antitone if `a ≤ b` implies `f b ≤ f a`. -/
def Antitone (f : α → β) : Prop :=
  ∀ ⦃a b⦄, a ≤ b → f b ≤ f a


/-- A function `f` is monotone on `s` if, for all `a, b ∈ s`, `a ≤ b` implies `f a ≤ f b`. -/
def MonotoneOn (f : α → β) (s : Set α) : Prop :=
  ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a ≤ b → f a ≤ f b


/-- A function `f` is antitone on `s` if, for all `a, b ∈ s`, `a ≤ b` implies `f b ≤ f a`. -/
def AntitoneOn (f : α → β) (s : Set α) : Prop :=
  ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a ≤ b → f b ≤ f a


/-- A function `f` is strictly monotone if `a < b` implies `f a < f b`. -/
def StrictMono (f : α → β) : Prop :=
  ∀ ⦃a b⦄, a < b → f a < f b


/-- A function `f` is strictly antitone if `a < b` implies `f b < f a`. -/
def StrictAnti (f : α → β) : Prop :=
  ∀ ⦃a b⦄, a < b → f b < f a


/-- A function `f` is strictly monotone on `s` if, for all `a, b ∈ s`, `a < b` implies
`f a < f b`. -/
def StrictMonoOn (f : α → β) (s : Set α) : Prop :=
  ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a < b → f a < f b


/-- A function `f` is strictly antitone on `s` if, for all `a, b ∈ s`, `a < b` implies
`f b < f a`. -/
def StrictAntiOn (f : α → β) (s : Set α) : Prop :=
  ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a < b → f b < f a


instance [i : Decidable (∀ a b, a ≤ b → f a ≤ f b)] : Decidable (Monotone f) := i

instance [i : Decidable (∀ a b, a ≤ b → f b ≤ f a)] : Decidable (Antitone f) := i


instance [i : Decidable (∀ a ∈ s, ∀ b ∈ s, a ≤ b → f a ≤ f b)] :
    Decidable (MonotoneOn f s) := i


instance [i : Decidable (∀ a ∈ s, ∀ b ∈ s, a ≤ b → f b ≤ f a)] :
    Decidable (AntitoneOn f s) := i


instance [i : Decidable (∀ a b, a < b → f a < f b)] : Decidable (StrictMono f) := i

instance [i : Decidable (∀ a b, a < b → f b < f a)] : Decidable (StrictAnti f) := i


instance [i : Decidable (∀ a ∈ s, ∀ b ∈ s, a < b → f a < f b)] :
    Decidable (StrictMonoOn f s) := i


instance [i : Decidable (∀ a ∈ s, ∀ b ∈ s, a < b → f b < f a)] :
    Decidable (StrictAntiOn f s) := i


lemma monotone_inclusion_le_le_of_le [Preorder α] {k j : α} (hkj : k ≤ j) :
    Monotone (fun ⟨i, hi⟩ => ⟨i, hi.trans hkj⟩ : { i // i ≤ k } → { i // i ≤ j}) :=
  fun _ _ h => h


lemma monotone_inclusion_lt_le_of_le [Preorder α] {k j : α} (hkj : k ≤ j) :
    Monotone (fun ⟨i, hi⟩ => ⟨i, hi.le.trans hkj⟩ : { i // i < k } → { i // i ≤ j}) :=
  fun _ _ h => h


lemma monotone_inclusion_lt_lt_of_le [Preorder α] {k j : α} (hkj : k ≤ j) :
    Monotone (fun ⟨i, hi⟩ => ⟨i, lt_of_lt_of_le hi hkj⟩ : { i // i < k } → { i // i < j}) :=
  fun _ _ h => h


@[simp]
theorem monotone_comp_ofDual_iff : Monotone (f ∘ ofDual) ↔ Antitone f :=
  forall_swap


@[simp]
theorem antitone_comp_ofDual_iff : Antitone (f ∘ ofDual) ↔ Monotone f :=
  forall_swap

-- Porting note:
-- Here (and below) without the type ascription, Lean is seeing through the
-- defeq `βᵒᵈ = β` and picking up the wrong `Preorder` instance.
-- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/logic.2Eequiv.2Ebasic.20mathlib4.23631/near/311744939

@[simp]
theorem monotone_toDual_comp_iff : Monotone (toDual ∘ f : α → βᵒᵈ) ↔ Antitone f :=
  Iff.rfl


@[simp]
theorem antitone_toDual_comp_iff : Antitone (toDual ∘ f : α → βᵒᵈ) ↔ Monotone f :=
  Iff.rfl


@[simp]
theorem monotoneOn_comp_ofDual_iff : MonotoneOn (f ∘ ofDual) s ↔ AntitoneOn f s :=
  forall₂_swap


@[simp]
theorem antitoneOn_comp_ofDual_iff : AntitoneOn (f ∘ ofDual) s ↔ MonotoneOn f s :=
  forall₂_swap


@[simp]
theorem monotoneOn_toDual_comp_iff : MonotoneOn (toDual ∘ f : α → βᵒᵈ) s ↔ AntitoneOn f s :=
  Iff.rfl


@[simp]
theorem antitoneOn_toDual_comp_iff : AntitoneOn (toDual ∘ f : α → βᵒᵈ) s ↔ MonotoneOn f s :=
  Iff.rfl


@[simp]
theorem strictMono_comp_ofDual_iff : StrictMono (f ∘ ofDual) ↔ StrictAnti f :=
  forall_swap


@[simp]
theorem strictAnti_comp_ofDual_iff : StrictAnti (f ∘ ofDual) ↔ StrictMono f :=
  forall_swap


@[simp]
theorem strictMono_toDual_comp_iff : StrictMono (toDual ∘ f : α → βᵒᵈ) ↔ StrictAnti f :=
  Iff.rfl


@[simp]
theorem strictAnti_toDual_comp_iff : StrictAnti (toDual ∘ f : α → βᵒᵈ) ↔ StrictMono f :=
  Iff.rfl


@[simp]
theorem strictMonoOn_comp_ofDual_iff : StrictMonoOn (f ∘ ofDual) s ↔ StrictAntiOn f s :=
  forall₂_swap


@[simp]
theorem strictAntiOn_comp_ofDual_iff : StrictAntiOn (f ∘ ofDual) s ↔ StrictMonoOn f s :=
  forall₂_swap


@[simp]
theorem strictMonoOn_toDual_comp_iff : StrictMonoOn (toDual ∘ f : α → βᵒᵈ) s ↔ StrictAntiOn f s :=
  Iff.rfl


@[simp]
theorem strictAntiOn_toDual_comp_iff : StrictAntiOn (toDual ∘ f : α → βᵒᵈ) s ↔ StrictMonoOn f s :=
  Iff.rfl


theorem monotone_dual_iff : Monotone (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) ↔ Monotone f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (Monotone (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDual …
  -/
  rw [monotone_toDual_comp_iff, antitone_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem antitone_dual_iff : Antitone (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) ↔ Antitone f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (Antitone (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDual …
  -/
  rw [antitone_toDual_comp_iff, monotone_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem monotoneOn_dual_iff : MonotoneOn (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) s ↔ MonotoneOn f s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    ⊢ Iff (MonotoneOn (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDu …
  -/
  rw [monotoneOn_toDual_comp_iff, antitoneOn_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem antitoneOn_dual_iff : AntitoneOn (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) s ↔ AntitoneOn f s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    ⊢ Iff (AntitoneOn (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDu …
  -/
  rw [antitoneOn_toDual_comp_iff, monotoneOn_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem strictMono_dual_iff : StrictMono (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) ↔ StrictMono f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (StrictMono (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDu …
  -/
  rw [strictMono_toDual_comp_iff, strictAnti_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem strictAnti_dual_iff : StrictAnti (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) ↔ StrictAnti f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (StrictAnti (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑OrderDu …
  -/
  rw [strictAnti_toDual_comp_iff, strictMono_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem strictMonoOn_dual_iff :
    StrictMonoOn (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) s ↔ StrictMonoOn f s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    ⊢ Iff (StrictMonoOn (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑Order …
  -/
  rw [strictMonoOn_toDual_comp_iff, strictAntiOn_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


theorem strictAntiOn_dual_iff :
    StrictAntiOn (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ) s ↔ StrictAntiOn f s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    ⊢ Iff (StrictAntiOn (Function.comp (⇑OrderDual.toDual) (Function.comp f ⇑Order …
  -/
  rw [strictAntiOn_toDual_comp_iff, strictMonoOn_comp_ofDual_iff]
  /-
    🎉 no goals
  -/


alias ⟨_, Monotone.dual_left⟩ := antitone_comp_ofDual_iff


alias ⟨_, Antitone.dual_left⟩ := monotone_comp_ofDual_iff


alias ⟨_, Monotone.dual_right⟩ := antitone_toDual_comp_iff


alias ⟨_, Antitone.dual_right⟩ := monotone_toDual_comp_iff


alias ⟨_, MonotoneOn.dual_left⟩ := antitoneOn_comp_ofDual_iff


alias ⟨_, AntitoneOn.dual_left⟩ := monotoneOn_comp_ofDual_iff


alias ⟨_, MonotoneOn.dual_right⟩ := antitoneOn_toDual_comp_iff


alias ⟨_, AntitoneOn.dual_right⟩ := monotoneOn_toDual_comp_iff


alias ⟨_, StrictMono.dual_left⟩ := strictAnti_comp_ofDual_iff


alias ⟨_, StrictAnti.dual_left⟩ := strictMono_comp_ofDual_iff


alias ⟨_, StrictMono.dual_right⟩ := strictAnti_toDual_comp_iff


alias ⟨_, StrictAnti.dual_right⟩ := strictMono_toDual_comp_iff


alias ⟨_, StrictMonoOn.dual_left⟩ := strictAntiOn_comp_ofDual_iff


alias ⟨_, StrictAntiOn.dual_left⟩ := strictMonoOn_comp_ofDual_iff


alias ⟨_, StrictMonoOn.dual_right⟩ := strictAntiOn_toDual_comp_iff


alias ⟨_, StrictAntiOn.dual_right⟩ := strictMonoOn_toDual_comp_iff


alias ⟨_, Monotone.dual⟩ := monotone_dual_iff


alias ⟨_, Antitone.dual⟩ := antitone_dual_iff


alias ⟨_, MonotoneOn.dual⟩ := monotoneOn_dual_iff


alias ⟨_, AntitoneOn.dual⟩ := antitoneOn_dual_iff


alias ⟨_, StrictMono.dual⟩ := strictMono_dual_iff


alias ⟨_, StrictAnti.dual⟩ := strictAnti_dual_iff


alias ⟨_, StrictMonoOn.dual⟩ := strictMonoOn_dual_iff


alias ⟨_, StrictAntiOn.dual⟩ := strictAntiOn_dual_iff


theorem StrictMono.wellFoundedLT [WellFoundedLT β] (hf : StrictMono f) : WellFoundedLT α :=
  Subrelation.isWellFounded (InvImage (· < ·) f) @hf


theorem StrictAnti.wellFoundedLT [WellFoundedGT β] (hf : StrictAnti f) : WellFoundedLT α :=
  StrictMono.wellFoundedLT (β := βᵒᵈ) hf


theorem StrictMono.wellFoundedGT [WellFoundedGT β] (hf : StrictMono f) : WellFoundedGT α :=
  StrictMono.wellFoundedLT (α := αᵒᵈ) (β := βᵒᵈ) (fun _ _ h ↦ hf h)


theorem StrictAnti.wellFoundedGT [WellFoundedLT β] (hf : StrictAnti f) : WellFoundedGT α :=
  StrictMono.wellFoundedLT (α := αᵒᵈ) (fun _ _ h ↦ hf h)


theorem Monotone.comp_le_comp_left
    [Preorder β] {f : β → α} {g h : γ → β} (hf : Monotone f) (le_gh : g ≤ h) :
    LE.le.{max w u} (f ∘ g) (f ∘ h) :=
  fun x ↦ hf (le_gh x)


theorem monotone_lam {f : α → β → γ} (hf : ∀ b, Monotone fun a ↦ f a b) : Monotone f :=
  fun _ _ h b ↦ hf b h


theorem monotone_app (f : β → α → γ) (b : β) (hf : Monotone fun a b ↦ f b a) : Monotone (f b) :=
  fun _ _ h ↦ hf h b


theorem antitone_lam {f : α → β → γ} (hf : ∀ b, Antitone fun a ↦ f a b) : Antitone f :=
  fun _ _ h b ↦ hf b h


theorem antitone_app (f : β → α → γ) (b : β) (hf : Antitone fun a b ↦ f b a) : Antitone (f b) :=
  fun _ _ h ↦ hf h b


theorem Function.monotone_eval {ι : Type u} {α : ι → Type v} [∀ i, Preorder (α i)] (i : ι) :
    Monotone (Function.eval i : (∀ i, α i) → α i) := fun _ _ H ↦ H i


theorem Monotone.imp (hf : Monotone f) (h : a ≤ b) : f a ≤ f b :=
  hf h


theorem Antitone.imp (hf : Antitone f) (h : a ≤ b) : f b ≤ f a :=
  hf h


theorem StrictMono.imp (hf : StrictMono f) (h : a < b) : f a < f b :=
  hf h


theorem StrictAnti.imp (hf : StrictAnti f) (h : a < b) : f b < f a :=
  hf h


protected theorem Monotone.monotoneOn (hf : Monotone f) (s : Set α) : MonotoneOn f s :=
  fun _ _ _ _ ↦ hf.imp


protected theorem Antitone.antitoneOn (hf : Antitone f) (s : Set α) : AntitoneOn f s :=
  fun _ _ _ _ ↦ hf.imp


@[simp] theorem monotoneOn_univ : MonotoneOn f Set.univ ↔ Monotone f :=
  ⟨fun h _ _ ↦ h trivial trivial, fun h ↦ h.monotoneOn _⟩


@[simp] theorem antitoneOn_univ : AntitoneOn f Set.univ ↔ Antitone f :=
  ⟨fun h _ _ ↦ h trivial trivial, fun h ↦ h.antitoneOn _⟩


protected theorem StrictMono.strictMonoOn (hf : StrictMono f) (s : Set α) : StrictMonoOn f s :=
  fun _ _ _ _ ↦ hf.imp


protected theorem StrictAnti.strictAntiOn (hf : StrictAnti f) (s : Set α) : StrictAntiOn f s :=
  fun _ _ _ _ ↦ hf.imp


@[simp] theorem strictMonoOn_univ : StrictMonoOn f Set.univ ↔ StrictMono f :=
  ⟨fun h _ _ ↦ h trivial trivial, fun h ↦ h.strictMonoOn _⟩


@[simp] theorem strictAntiOn_univ : StrictAntiOn f Set.univ ↔ StrictAnti f :=
  ⟨fun h _ _ ↦ h trivial trivial, fun h ↦ h.strictAntiOn _⟩


theorem Monotone.strictMono_of_injective (h₁ : Monotone f) (h₂ : Injective f) : StrictMono f :=
  fun _ _ h ↦ (h₁ h.le).lt_of_ne fun H ↦ h.ne <| h₂ H


theorem Antitone.strictAnti_of_injective (h₁ : Antitone f) (h₂ : Injective f) : StrictAnti f :=
  fun _ _ h ↦ (h₁ h.le).lt_of_ne fun H ↦ h.ne <| h₂ H.symm


theorem monotone_iff_forall_lt : Monotone f ↔ ∀ ⦃a b⦄, a < b → f a ≤ f b :=
  forall₂_congr fun _ _ ↦
    ⟨fun hf h ↦ hf h.le, fun hf h ↦ h.eq_or_lt.elim (fun H ↦ (congr_arg _ H).le) hf⟩


theorem antitone_iff_forall_lt : Antitone f ↔ ∀ ⦃a b⦄, a < b → f b ≤ f a :=
  forall₂_congr fun _ _ ↦
    ⟨fun hf h ↦ hf h.le, fun hf h ↦ h.eq_or_lt.elim (fun H ↦ (congr_arg _ H).ge) hf⟩


theorem monotoneOn_iff_forall_lt :
    MonotoneOn f s ↔ ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a < b → f a ≤ f b :=
  ⟨fun hf _ ha _ hb h ↦ hf ha hb h.le,
   fun hf _ ha _ hb h ↦ h.eq_or_lt.elim (fun H ↦ (congr_arg _ H).le) (hf ha hb)⟩


theorem antitoneOn_iff_forall_lt :
    AntitoneOn f s ↔ ∀ ⦃a⦄ (_ : a ∈ s) ⦃b⦄ (_ : b ∈ s), a < b → f b ≤ f a :=
  ⟨fun hf _ ha _ hb h ↦ hf ha hb h.le,
   fun hf _ ha _ hb h ↦ h.eq_or_lt.elim (fun H ↦ (congr_arg _ H).ge) (hf ha hb)⟩

-- `Preorder α` isn't strong enough: if the preorder on `α` is an equivalence relation,
-- then `StrictMono f` is vacuously true.

protected theorem StrictMonoOn.monotoneOn (hf : StrictMonoOn f s) : MonotoneOn f s :=
  monotoneOn_iff_forall_lt.2 fun _ ha _ hb h ↦ (hf ha hb h).le


protected theorem StrictAntiOn.antitoneOn (hf : StrictAntiOn f s) : AntitoneOn f s :=
  antitoneOn_iff_forall_lt.2 fun _ ha _ hb h ↦ (hf ha hb h).le


protected theorem StrictMono.monotone (hf : StrictMono f) : Monotone f :=
  monotone_iff_forall_lt.2 fun _ _ h ↦ (hf h).le


protected theorem StrictAnti.antitone (hf : StrictAnti f) : Antitone f :=
  antitone_iff_forall_lt.2 fun _ _ h ↦ (hf h).le


protected theorem monotone [Subsingleton α] (f : α → β) : Monotone f :=
  fun _ _ _ ↦ (congr_arg _ <| Subsingleton.elim _ _).le


protected theorem antitone [Subsingleton α] (f : α → β) : Antitone f :=
  fun _ _ _ ↦ (congr_arg _ <| Subsingleton.elim _ _).le


theorem monotone' [Subsingleton β] (f : α → β) : Monotone f :=
  fun _ _ _ ↦ (Subsingleton.elim _ _).le


theorem antitone' [Subsingleton β] (f : α → β) : Antitone f :=
  fun _ _ _ ↦ (Subsingleton.elim _ _).le


protected theorem strictMono [Subsingleton α] (f : α → β) : StrictMono f :=
  fun _ _ h ↦ (h.ne <| Subsingleton.elim _ _).elim


protected theorem strictAnti [Subsingleton α] (f : α → β) : StrictAnti f :=
  fun _ _ h ↦ (h.ne <| Subsingleton.elim _ _).elim


theorem monotone_id [Preorder α] : Monotone (id : α → α) := fun _ _ ↦ id


theorem monotoneOn_id [Preorder α] {s : Set α} : MonotoneOn id s := fun _ _ _ _ ↦ id


theorem strictMono_id [Preorder α] : StrictMono (id : α → α) := fun _ _ ↦ id


theorem strictMonoOn_id [Preorder α] {s : Set α} : StrictMonoOn id s := fun _ _ _ _ ↦ id


theorem monotone_const [Preorder α] [Preorder β] {c : β} : Monotone fun _ : α ↦ c :=
  fun _ _ _ ↦ le_rfl


theorem monotoneOn_const [Preorder α] [Preorder β] {c : β} {s : Set α} :
    MonotoneOn (fun _ : α ↦ c) s :=
  fun _ _ _ _ _ ↦ le_rfl


theorem antitone_const [Preorder α] [Preorder β] {c : β} : Antitone fun _ : α ↦ c :=
  fun _ _ _ ↦ le_refl c


theorem antitoneOn_const [Preorder α] [Preorder β] {c : β} {s : Set α} :
    AntitoneOn (fun _ : α ↦ c) s :=
  fun _ _ _ _ _ ↦ le_rfl


theorem strictMono_of_le_iff_le [Preorder α] [Preorder β] {f : α → β}
    (h : ∀ x y, x ≤ y ↔ f x ≤ f y) : StrictMono f :=
  fun _ _ ↦ (lt_iff_lt_of_le_iff_le' (h _ _) (h _ _)).1


theorem strictAnti_of_le_iff_le [Preorder α] [Preorder β] {f : α → β}
    (h : ∀ x y, x ≤ y ↔ f y ≤ f x) : StrictAnti f :=
  fun _ _ ↦ (lt_iff_lt_of_le_iff_le' (h _ _) (h _ _)).1

-- Porting note: mathlib3 proof uses `contrapose` tactic

theorem injective_of_lt_imp_ne [LinearOrder α] {f : α → β} (h : ∀ x y, x < y → f x ≠ f y) :
    Injective f := by
  /-
    α : Type u
    β : Type v
    inst✝ : LinearOrder α
    f : α → β
    h : ∀ (x y : α), LT.lt x y → Ne (f x) (f y)
    ⊢ Function.Injective f
  -/
  intro x y hf
  /-
    α : Type u
    β : Type v
    inst✝ : LinearOrder α
    f : α → β
    h : ∀ (x y : α), LT.lt x y → Ne (f x) (f y)
    x y : α
    hf : Eq (f x) (f y)
    ⊢ Eq x y
  -/
  rcases lt_trichotomy x y with (hxy | rfl | hxy)
    /-
      case inl
      α : Type u
      β : Type v
      inst✝ : LinearOrder α
      f : α → β
      h : ∀ (x y : α), LT.lt x y → Ne (f x) (f y)
      x y : α
      hf : Eq (f x) (f y)
      hxy : LT.lt x y
      ⊢ Eq x y
    -/
  · exact absurd hf <| h _ _ hxy
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u
      β : Type v
      inst✝ : LinearOrder α
      f : α → β
      h : ∀ (x y : α), LT.lt x y → Ne (f x) (f y)
      x : α
      hf : Eq (f x) (f x)
      ⊢ Eq x x
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u
      β : Type v
      inst✝ : LinearOrder α
      f : α → β
      h : ∀ (x y : α), LT.lt x y → Ne (f x) (f y)
      x y : α
      hf : Eq (f x) (f y)
      hxy : LT.lt y x
      ⊢ Eq x y
    -/
  · exact absurd hf.symm <| h _ _ hxy
    /-
      🎉 no goals
    -/


theorem injective_of_le_imp_le [PartialOrder α] [Preorder β] (f : α → β)
    (h : ∀ {x y}, f x ≤ f y → x ≤ y) : Injective f :=
  fun _ _ hxy ↦ (h hxy.le).antisymm (h hxy.ge)


theorem StrictMono.isMax_of_apply (hf : StrictMono f) (ha : IsMax (f a)) : IsMax a :=
  of_not_not fun h ↦
    let ⟨_, hb⟩ := not_isMax_iff.1 h
    (hf hb).not_isMax ha


theorem StrictMono.isMin_of_apply (hf : StrictMono f) (ha : IsMin (f a)) : IsMin a :=
  of_not_not fun h ↦
    let ⟨_, hb⟩ := not_isMin_iff.1 h
    (hf hb).not_isMin ha


theorem StrictAnti.isMax_of_apply (hf : StrictAnti f) (ha : IsMin (f a)) : IsMax a :=
  of_not_not fun h ↦
    let ⟨_, hb⟩ := not_isMax_iff.1 h
    (hf hb).not_isMin ha


theorem StrictAnti.isMin_of_apply (hf : StrictAnti f) (ha : IsMax (f a)) : IsMin a :=
  of_not_not fun h ↦
    let ⟨_, hb⟩ := not_isMin_iff.1 h
    (hf hb).not_isMax ha


lemma StrictMono.add_le_nat {f : ℕ → ℕ} (hf : StrictMono f) (m n : ℕ) : m + f n ≤ f (m + n)  := by
  /-
    f : Nat → Nat
    hf : StrictMono f
    m n : Nat
    ⊢ LE.le (HAdd.hAdd m (f n)) (f (HAdd.hAdd m n))
  -/
  rw [Nat.add_comm m, Nat.add_comm m]
  induction m with
  | zero => rw [Nat.add_zero, Nat.add_zero]
  | succ m ih =>
    rw [← Nat.add_assoc, ← Nat.add_assoc, Nat.succ_le]
    exact ih.trans_lt (hf (n + m).lt_succ_self)


protected theorem StrictMono.ite' (hf : StrictMono f) (hg : StrictMono g) {p : α → Prop}
    [DecidablePred p]
    (hp : ∀ ⦃x y⦄, x < y → p y → p x) (hfg : ∀ ⦃x y⦄, p x → ¬p y → x < y → f x < g y) :
    StrictMono fun x ↦ if p x then f x else g x := by
  /-
    α : Type u
    β : Type v
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    f g : α → β
    hf : StrictMono f
    hg : StrictMono g
    p : α → Prop
    inst✝ : DecidablePred p
    hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
    hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
    ⊢ StrictMono fun x => ite (p x) (f x) (g x)
  -/
  intro x y h
  /-
    α : Type u
    β : Type v
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    f g : α → β
    hf : StrictMono f
    hg : StrictMono g
    p : α → Prop
    inst✝ : DecidablePred p
    hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
    hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
    x y : α
    h : LT.lt x y
    ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
  -/
  by_cases hy : p y
    /-
      case pos
      α : Type u
      β : Type v
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      f g : α → β
      hf : StrictMono f
      hg : StrictMono g
      p : α → Prop
      inst✝ : DecidablePred p
      hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
      hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
      x y : α
      h : LT.lt x y
      hy : p y
      ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
    -/
  · have hx : p x := hp h hy
    /-
      case pos
      α : Type u
      β : Type v
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      f g : α → β
      hf : StrictMono f
      hg : StrictMono g
      p : α → Prop
      inst✝ : DecidablePred p
      hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
      hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
      x y : α
      h : LT.lt x y
      hy : p y
      hx : p x
      ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
    -/
    simpa [hx, hy] using hf h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    β : Type v
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    f g : α → β
    hf : StrictMono f
    hg : StrictMono g
    p : α → Prop
    inst✝ : DecidablePred p
    hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
    hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
    x y : α
    h : LT.lt x y
    hy : Not (p y)
    ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
  -/
  by_cases hx : p x
    /-
      case pos
      α : Type u
      β : Type v
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      f g : α → β
      hf : StrictMono f
      hg : StrictMono g
      p : α → Prop
      inst✝ : DecidablePred p
      hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
      hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
      x y : α
      h : LT.lt x y
      hy : Not (p y)
      hx : p x
      ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
    -/
  · simpa [hx, hy] using hfg hx hy h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      β : Type v
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      f g : α → β
      hf : StrictMono f
      hg : StrictMono g
      p : α → Prop
      inst✝ : DecidablePred p
      hp : ∀ ⦃x y : α⦄, LT.lt x y → p y → p x
      hfg : ∀ ⦃x y : α⦄, p x → Not (p y) → LT.lt x y → LT.lt (f x) (g y)
      x y : α
      h : LT.lt x y
      hy : Not (p y)
      hx : Not (p x)
      ⊢ LT.lt ((fun x => ite (p x) (f x) (g x)) x) ((fun x => ite (p x) (f x) (g x)) …
    -/
  · simpa [hx, hy] using hg h
    /-
      🎉 no goals
    -/


protected theorem StrictMono.ite (hf : StrictMono f) (hg : StrictMono g) {p : α → Prop}
    [DecidablePred p] (hp : ∀ ⦃x y⦄, x < y → p y → p x) (hfg : ∀ x, f x ≤ g x) :
    StrictMono fun x ↦ if p x then f x else g x :=
  (hf.ite' hg hp) fun _ y _ _ h ↦ (hf h).trans_le (hfg y)

-- Porting note: `Strict*.dual_right` dot notation is not working here for some reason

protected theorem StrictAnti.ite' (hf : StrictAnti f) (hg : StrictAnti g) {p : α → Prop}
    [DecidablePred p]
    (hp : ∀ ⦃x y⦄, x < y → p y → p x) (hfg : ∀ ⦃x y⦄, p x → ¬p y → x < y → g y < f x) :
    StrictAnti fun x ↦ if p x then f x else g x :=
  StrictMono.ite' (StrictAnti.dual_right hf) (StrictAnti.dual_right hg) hp hfg


protected theorem StrictAnti.ite (hf : StrictAnti f) (hg : StrictAnti g) {p : α → Prop}
    [DecidablePred p] (hp : ∀ ⦃x y⦄, x < y → p y → p x) (hfg : ∀ x, g x ≤ f x) :
    StrictAnti fun x ↦ if p x then f x else g x :=
  (hf.ite' hg hp) fun _ y _ _ h ↦ (hfg y).trans_lt (hf h)


protected theorem Monotone.comp (hg : Monotone g) (hf : Monotone f) : Monotone (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


theorem Monotone.comp_antitone (hg : Monotone g) (hf : Antitone f) : Antitone (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


protected theorem Antitone.comp (hg : Antitone g) (hf : Antitone f) : Monotone (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


theorem Antitone.comp_monotone (hg : Antitone g) (hf : Monotone f) : Antitone (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


protected theorem Monotone.iterate {f : α → α} (hf : Monotone f) (n : ℕ) : Monotone f^[n] :=
  Nat.recOn n monotone_id fun _ h ↦ h.comp hf


protected theorem Monotone.comp_monotoneOn (hg : Monotone g) (hf : MonotoneOn f s) :
    MonotoneOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


theorem Monotone.comp_antitoneOn (hg : Monotone g) (hf : AntitoneOn f s) : AntitoneOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


protected theorem Antitone.comp_antitoneOn (hg : Antitone g) (hf : AntitoneOn f s) :
    MonotoneOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


theorem Antitone.comp_monotoneOn (hg : Antitone g) (hf : MonotoneOn f s) : AntitoneOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


protected theorem StrictMono.comp (hg : StrictMono g) (hf : StrictMono f) : StrictMono (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


theorem StrictMono.comp_strictAnti (hg : StrictMono g) (hf : StrictAnti f) : StrictAnti (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


protected theorem StrictAnti.comp (hg : StrictAnti g) (hf : StrictAnti f) : StrictMono (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


theorem StrictAnti.comp_strictMono (hg : StrictAnti g) (hf : StrictMono f) : StrictAnti (g ∘ f) :=
  fun _ _ h ↦ hg (hf h)


protected theorem StrictMono.iterate {f : α → α} (hf : StrictMono f) (n : ℕ) : StrictMono f^[n] :=
  Nat.recOn n strictMono_id fun _ h ↦ h.comp hf


protected theorem StrictMono.comp_strictMonoOn (hg : StrictMono g) (hf : StrictMonoOn f s) :
    StrictMonoOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


theorem StrictMono.comp_strictAntiOn (hg : StrictMono g) (hf : StrictAntiOn f s) :
    StrictAntiOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


protected theorem StrictAnti.comp_strictAntiOn (hg : StrictAnti g) (hf : StrictAntiOn f s) :
    StrictMonoOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


theorem StrictAnti.comp_strictMonoOn (hg : StrictAnti g) (hf : StrictMonoOn f s) :
    StrictAntiOn (g ∘ f) s :=
  fun _ ha _ hb h ↦ hg (hf ha hb h)


theorem foldl_monotone [Preorder α] {f : α → β → α} (H : ∀ b, Monotone fun a ↦ f a b)
    (l : List β) : Monotone fun a ↦ l.foldl f a :=
  List.recOn l (fun _ _ ↦ id) fun _ _ hl _ _ h ↦ hl (H _ h)


theorem foldr_monotone [Preorder β] {f : α → β → β} (H : ∀ a, Monotone (f a)) (l : List α) :
    Monotone fun b ↦ l.foldr f b := fun _ _ h ↦ List.recOn l h fun i _ hl ↦ H i hl


theorem foldl_strictMono [Preorder α] {f : α → β → α} (H : ∀ b, StrictMono fun a ↦ f a b)
    (l : List β) : StrictMono fun a ↦ l.foldl f a :=
  List.recOn l (fun _ _ ↦ id) fun _ _ hl _ _ h ↦ hl (H _ h)


theorem foldr_strictMono [Preorder β] {f : α → β → β} (H : ∀ a, StrictMono (f a)) (l : List α) :
    StrictMono fun b ↦ l.foldr f b := fun _ _ h ↦ List.recOn l h fun i _ hl ↦ H i hl


theorem Monotone.reflect_lt (hf : Monotone f) {a b : α} (h : f a < f b) : a < b :=
  lt_of_not_ge fun h' ↦ h.not_le (hf h')


theorem Antitone.reflect_lt (hf : Antitone f) {a b : α} (h : f a < f b) : b < a :=
  lt_of_not_ge fun h' ↦ h.not_le (hf h')


theorem MonotoneOn.reflect_lt (hf : MonotoneOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s)
    (h : f a < f b) : a < b :=
  lt_of_not_ge fun h' ↦ h.not_le <| hf hb ha h'


theorem AntitoneOn.reflect_lt (hf : AntitoneOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s)
    (h : f a < f b) : b < a :=
  lt_of_not_ge fun h' ↦ h.not_le <| hf ha hb h'


theorem StrictMonoOn.le_iff_le (hf : StrictMonoOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a ≤ f b ↔ a ≤ b :=
  ⟨fun h ↦ le_of_not_gt fun h' ↦ (hf hb ha h').not_le h, fun h ↦
    h.lt_or_eq_dec.elim (fun h' ↦ (hf ha hb h').le) fun h' ↦ h' ▸ le_rfl⟩


theorem StrictAntiOn.le_iff_le (hf : StrictAntiOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a ≤ f b ↔ b ≤ a :=
  hf.dual_right.le_iff_le hb ha


theorem StrictMonoOn.eq_iff_eq (hf : StrictMonoOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a = f b ↔ a = b :=
  ⟨fun h ↦ le_antisymm ((hf.le_iff_le ha hb).mp h.le) ((hf.le_iff_le hb ha).mp h.ge), by
    /-
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s : Set α
      hf : StrictMonoOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      ⊢ Eq a b → Eq (f a) (f b)
    -/
    rintro rfl
    /-
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s : Set α
      hf : StrictMonoOn f s
      a : α
      ha hb : Membership.mem s a
      ⊢ Eq (f a) (f a)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem StrictAntiOn.eq_iff_eq (hf : StrictAntiOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a = f b ↔ b = a :=
  (hf.dual_right.eq_iff_eq ha hb).trans eq_comm


theorem StrictMonoOn.lt_iff_lt (hf : StrictMonoOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a < f b ↔ a < b := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    hf : StrictMonoOn f s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Iff (LT.lt (f a) (f b)) (LT.lt a b)
  -/
  rw [lt_iff_le_not_le, lt_iff_le_not_le, hf.le_iff_le ha hb, hf.le_iff_le hb ha]
  /-
    🎉 no goals
  -/


theorem StrictAntiOn.lt_iff_lt (hf : StrictAntiOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    f a < f b ↔ b < a :=
  hf.dual_right.lt_iff_lt hb ha


theorem StrictMono.le_iff_le (hf : StrictMono f) {a b : α} : f a ≤ f b ↔ a ≤ b :=
  (hf.strictMonoOn Set.univ).le_iff_le trivial trivial


theorem StrictAnti.le_iff_le (hf : StrictAnti f) {a b : α} : f a ≤ f b ↔ b ≤ a :=
  (hf.strictAntiOn Set.univ).le_iff_le trivial trivial


theorem StrictMono.lt_iff_lt (hf : StrictMono f) {a b : α} : f a < f b ↔ a < b :=
  (hf.strictMonoOn Set.univ).lt_iff_lt trivial trivial


theorem StrictAnti.lt_iff_lt (hf : StrictAnti f) {a b : α} : f a < f b ↔ b < a :=
  (hf.strictAntiOn Set.univ).lt_iff_lt trivial trivial


protected theorem StrictMonoOn.compares (hf : StrictMonoOn f s) {a b : α} (ha : a ∈ s)
    (hb : b ∈ s) : ∀ {o : Ordering}, o.Compares (f a) (f b) ↔ o.Compares a b
  | Ordering.lt => hf.lt_iff_lt ha hb
  | Ordering.eq => ⟨fun h ↦ ((hf.le_iff_le ha hb).1 h.le).antisymm
                      ((hf.le_iff_le hb ha).1 h.symm.le), congr_arg _⟩
  | Ordering.gt => hf.lt_iff_lt hb ha


protected theorem StrictAntiOn.compares (hf : StrictAntiOn f s) {a b : α} (ha : a ∈ s)
    (hb : b ∈ s) {o : Ordering} : o.Compares (f a) (f b) ↔ o.Compares b a :=
  toDual_compares_toDual.trans <| hf.dual_right.compares hb ha


protected theorem StrictMono.compares (hf : StrictMono f) {a b : α} {o : Ordering} :
    o.Compares (f a) (f b) ↔ o.Compares a b :=
  (hf.strictMonoOn Set.univ).compares trivial trivial


protected theorem StrictAnti.compares (hf : StrictAnti f) {a b : α} {o : Ordering} :
    o.Compares (f a) (f b) ↔ o.Compares b a :=
  (hf.strictAntiOn Set.univ).compares trivial trivial


theorem StrictMono.injective (hf : StrictMono f) : Injective f :=
  fun x y h ↦ show Compares eq x y from hf.compares.1 h


theorem StrictAnti.injective (hf : StrictAnti f) : Injective f :=
  fun x y h ↦ show Compares eq x y from hf.compares.1 h.symm


theorem StrictMono.maximal_of_maximal_image (hf : StrictMono f) {a} (hmax : ∀ p, p ≤ f a) (x : α) :
    x ≤ a :=
  hf.le_iff_le.mp (hmax (f x))


theorem StrictMono.minimal_of_minimal_image (hf : StrictMono f) {a} (hmin : ∀ p, f a ≤ p) (x : α) :
    a ≤ x :=
  hf.le_iff_le.mp (hmin (f x))


theorem StrictAnti.minimal_of_maximal_image (hf : StrictAnti f) {a} (hmax : ∀ p, p ≤ f a) (x : α) :
    a ≤ x :=
  hf.le_iff_le.mp (hmax (f x))


theorem StrictAnti.maximal_of_minimal_image (hf : StrictAnti f) {a} (hmin : ∀ p, f a ≤ p) (x : α) :
    x ≤ a :=
  hf.le_iff_le.mp (hmin (f x))


theorem Monotone.strictMono_iff_injective (hf : Monotone f) : StrictMono f ↔ Injective f :=
  ⟨fun h ↦ h.injective, hf.strictMono_of_injective⟩


theorem Antitone.strictAnti_iff_injective (hf : Antitone f) : StrictAnti f ↔ Injective f :=
  ⟨fun h ↦ h.injective, hf.strictAnti_of_injective⟩


/-- If a monotone function is equal at two points, it is equal between all of them -/
theorem Monotone.eq_of_le_of_le {a₁ a₂ : α} (h_mon : Monotone f) (h_fa : f a₁ = f a₂) {i : α}
    (h₁ : a₁ ≤ i) (h₂ : i ≤ a₂) : f i = f a₁ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    a₁ a₂ : α
    h_mon : Monotone f
    h_fa : Eq (f a₁) (f a₂)
    i : α
    h₁ : LE.le a₁ i
    h₂ : LE.le i a₂
    ⊢ Eq (f i) (f a₁)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      a₁ a₂ : α
      h_mon : Monotone f
      h_fa : Eq (f a₁) (f a₂)
      i : α
      h₁ : LE.le a₁ i
      h₂ : LE.le i a₂
      ⊢ LE.le (f i) (f a₁)
    -/
  · rw [h_fa]; exact h_mon h₂
               /-
                 🎉 no goals
               -/
    /-
      case a
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      a₁ a₂ : α
      h_mon : Monotone f
      h_fa : Eq (f a₁) (f a₂)
      i : α
      h₁ : LE.le a₁ i
      h₂ : LE.le i a₂
      ⊢ LE.le (f a₁) (f i)
    -/
  · exact h_mon h₁
    /-
      🎉 no goals
    -/


/-- If an antitone function is equal at two points, it is equal between all of them -/
theorem Antitone.eq_of_le_of_le {a₁ a₂ : α} (h_anti : Antitone f) (h_fa : f a₁ = f a₂) {i : α}
    (h₁ : a₁ ≤ i) (h₂ : i ≤ a₂) : f i = f a₁ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    a₁ a₂ : α
    h_anti : Antitone f
    h_fa : Eq (f a₁) (f a₂)
    i : α
    h₁ : LE.le a₁ i
    h₂ : LE.le i a₂
    ⊢ Eq (f i) (f a₁)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      a₁ a₂ : α
      h_anti : Antitone f
      h_fa : Eq (f a₁) (f a₂)
      i : α
      h₁ : LE.le a₁ i
      h₂ : LE.le i a₂
      ⊢ LE.le (f i) (f a₁)
    -/
  · exact h_anti h₁
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      a₁ a₂ : α
      h_anti : Antitone f
      h_fa : Eq (f a₁) (f a₂)
      i : α
      h₁ : LE.le a₁ i
      h₂ : LE.le i a₂
      ⊢ LE.le (f a₁) (f i)
    -/
  · rw [h_fa]; exact h_anti h₂
               /-
                 🎉 no goals
               -/


/-- A function between linear orders which is neither monotone nor antitone makes a dent upright or
downright. -/
lemma not_monotone_not_antitone_iff_exists_le_le :
    ¬ Monotone f ∧ ¬ Antitone f ↔
      ∃ a b c, a ≤ b ∧ b ≤ c ∧ ((f a < f b ∧ f c < f b) ∨ (f b < f a ∧ f b < f c)) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Iff (And (Not (Monotone f)) (Not (Antitone f))) (Exists fun a => Exists fun  …
  -/
  simp_rw [Monotone, Antitone, not_forall, not_le]
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Iff (And (Exists fun x => Exists fun x_1 => Exists fun h => LT.lt (f x_1) (f …
  -/
  refine Iff.symm ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      ⊢ (Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le …
    -/
  · rintro ⟨a, b, c, hab, hbc, ⟨hfab, hfcb⟩ | ⟨hfba, hfbc⟩⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.inl.intro
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b c : α
      hab : LE.le a b
      hbc : LE.le b c
      hfab : LT.lt (f a) (f b)
      hfcb : LT.lt (f c) (f b)
      ⊢ And (Exists fun x => Exists fun x_1 => Exists fun h => LT.lt (f x_1) (f x))  …
    -/
    exacts [⟨⟨_, _, hbc, hfcb⟩, _, _, hab, hfab⟩, ⟨⟨_, _, hab, hfba⟩, _, _, hbc, hfbc⟩]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ And (Exists fun x => Exists fun x_1 => Exists fun h => LT.lt (f x_1) (f x))  …
  -/
  rintro ⟨⟨a, b, hab, hfba⟩, c, d, hcd, hfcd⟩
  /-
    case refine_2.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    a b : α
    hab : LE.le a b
    hfba : LT.lt (f b) (f a)
    c d : α
    hcd : LE.le c d
    hfcd : LT.lt (f c) (f d)
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
  -/
  obtain hda | had := le_total d a
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inl
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      hda : LE.le d a
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
  · obtain hfad | hfda := le_total (f a) (f d)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inl.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        hda : LE.le d a
        hfad : LE.le (f a) (f d)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨c, d, b, hcd, hda.trans hab, Or.inl ⟨hfcd, hfba.trans_le hfad⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inl.inr
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        hda : LE.le d a
        hfda : LE.le (f d) (f a)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨c, a, b, hcd.trans hda, hab, Or.inl ⟨hfcd.trans_le hfda, hfba⟩⟩
      /-
        🎉 no goals
      -/
  /-
    case refine_2.intro.intro.intro.intro.intro.intro.intro.inr
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    a b : α
    hab : LE.le a b
    hfba : LT.lt (f b) (f a)
    c d : α
    hcd : LE.le c d
    hfcd : LT.lt (f c) (f d)
    had : LE.le a d
    ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
  -/
  obtain hac | hca := le_total a c
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hac : LE.le a c
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
  · obtain hfdb | hfbd := le_or_lt (f d) (f b)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hac : LE.le a c
        hfdb : LE.le (f d) (f b)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, c, d, hac, hcd, Or.inr ⟨hfcd.trans <| hfdb.trans_lt hfba, hfcd⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inr
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hac : LE.le a c
      hfbd : LT.lt (f b) (f d)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
    obtain hfca | hfac := lt_or_le (f c) (f a)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inr.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hac : LE.le a c
        hfbd : LT.lt (f b) (f d)
        hfca : LT.lt (f c) (f a)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, c, d, hac, hcd, Or.inr ⟨hfca, hfcd⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inr.inr
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hac : LE.le a c
      hfbd : LT.lt (f b) (f d)
      hfac : LE.le (f a) (f c)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
    obtain hbd | hdb := le_total b d
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inr.inr.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hac : LE.le a c
        hfbd : LT.lt (f b) (f d)
        hfac : LE.le (f a) (f c)
        hbd : LE.le b d
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, b, d, hab, hbd, Or.inr ⟨hfba, hfbd⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inl.inr.inr.inr
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hac : LE.le a c
        hfbd : LT.lt (f b) (f d)
        hfac : LE.le (f a) (f c)
        hdb : LE.le d b
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, d, b, had, hdb, Or.inl ⟨hfac.trans_lt hfcd, hfbd⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hca : LE.le c a
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
  · obtain hfdb | hfbd := le_or_lt (f d) (f b)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hca : LE.le c a
        hfdb : LE.le (f d) (f b)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨c, a, b, hca, hab, Or.inl ⟨hfcd.trans <| hfdb.trans_lt hfba, hfba⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inr
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hca : LE.le c a
      hfbd : LT.lt (f b) (f d)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
    obtain hfca | hfac := lt_or_le (f c) (f a)
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inr.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hca : LE.le c a
        hfbd : LT.lt (f b) (f d)
        hfca : LT.lt (f c) (f a)
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨c, a, b, hca, hab, Or.inl ⟨hfca, hfba⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inr.inr
      α : Type u
      β : Type v
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b : α
      hab : LE.le a b
      hfba : LT.lt (f b) (f a)
      c d : α
      hcd : LE.le c d
      hfcd : LT.lt (f c) (f d)
      had : LE.le a d
      hca : LE.le c a
      hfbd : LT.lt (f b) (f d)
      hfac : LE.le (f a) (f c)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
    -/
    obtain hbd | hdb := le_total b d
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inr.inr.inl
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hca : LE.le c a
        hfbd : LT.lt (f b) (f d)
        hfac : LE.le (f a) (f c)
        hbd : LE.le b d
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, b, d, hab, hbd, Or.inr ⟨hfba, hfbd⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.intro.intro.inr.inr.inr.inr.inr
        α : Type u
        β : Type v
        inst✝¹ : LinearOrder α
        inst✝ : LinearOrder β
        f : α → β
        a b : α
        hab : LE.le a b
        hfba : LT.lt (f b) (f a)
        c d : α
        hcd : LE.le c d
        hfcd : LT.lt (f c) (f d)
        had : LE.le a d
        hca : LE.le c a
        hfbd : LT.lt (f b) (f d)
        hfac : LE.le (f a) (f c)
        hdb : LE.le d b
        ⊢ Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le  …
      -/
    · exact ⟨a, d, b, had, hdb, Or.inl ⟨hfac.trans_lt hfcd, hfbd⟩⟩
      /-
        🎉 no goals
      -/


/-- A function between linear orders which is neither monotone nor antitone makes a dent upright or
downright. -/
lemma not_monotone_not_antitone_iff_exists_lt_lt :
    ¬ Monotone f ∧ ¬ Antitone f ↔ ∃ a b c, a < b ∧ b < c ∧
    (f a < f b ∧ f c < f b ∨ f b < f a ∧ f b < f c) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Iff (And (Not (Monotone f)) (Not (Antitone f))) (Exists fun a => Exists fun  …
  -/
  simp_rw [not_monotone_not_antitone_iff_exists_le_le, ← and_assoc]
  refine exists₃_congr (fun a b c ↦ and_congr_left <|
    fun h ↦ (Ne.le_iff_lt ?_).and <| Ne.le_iff_lt ?_) <;>
   /-
     case refine_1
     α : Type u
     β : Type v
     inst✝¹ : LinearOrder α
     inst✝ : LinearOrder β
     f : α → β
     a b c : α
     h : Or (And (LT.lt (f a) (f b)) (LT.lt (f c) (f b))) (And (LT.lt (f b) (f a))  …
     ⊢ Ne a b
   -/
               /-
                 🎉 no goals
               -/
  (rintro rfl; simp at h)
               /-
                 🎉 no goals
               -/


theorem StrictMonoOn.cmp_map_eq (hf : StrictMonoOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    cmp (f x) (f y) = cmp x y :=
  ((hf.compares hx hy).2 (cmp_compares x y)).cmp_eq


theorem StrictMono.cmp_map_eq (hf : StrictMono f) (x y : α) : cmp (f x) (f y) = cmp x y :=
  (hf.strictMonoOn Set.univ).cmp_map_eq trivial trivial


theorem StrictAntiOn.cmp_map_eq (hf : StrictAntiOn f s) (hx : x ∈ s) (hy : y ∈ s) :
    cmp (f x) (f y) = cmp y x :=
  hf.dual_right.cmp_map_eq hy hx


theorem StrictAnti.cmp_map_eq (hf : StrictAnti f) (x y : α) : cmp (f x) (f y) = cmp y x :=
  (hf.strictAntiOn Set.univ).cmp_map_eq trivial trivial


theorem Nat.rel_of_forall_rel_succ_of_le_of_lt (r : β → β → Prop) [IsTrans β r] {f : ℕ → β} {a : ℕ}
    (h : ∀ n, a ≤ n → r (f n) (f (n + 1))) ⦃b c : ℕ⦄ (hab : a ≤ b) (hbc : b < c) :
    r (f b) (f c) := by
  induction hbc with
  | refl => exact h _ hab
  | step b_lt_k r_b_k => exact _root_.trans r_b_k (h _ (hab.trans_lt b_lt_k).le)


theorem Nat.rel_of_forall_rel_succ_of_le_of_le (r : β → β → Prop) [IsRefl β r] [IsTrans β r]
    {f : ℕ → β} {a : ℕ} (h : ∀ n, a ≤ n → r (f n) (f (n + 1)))
    ⦃b c : ℕ⦄ (hab : a ≤ b) (hbc : b ≤ c) : r (f b) (f c) :=
  hbc.eq_or_lt.elim (fun h ↦ h ▸ refl _) (Nat.rel_of_forall_rel_succ_of_le_of_lt r h hab)


theorem Nat.rel_of_forall_rel_succ_of_lt (r : β → β → Prop) [IsTrans β r] {f : ℕ → β}
    (h : ∀ n, r (f n) (f (n + 1))) ⦃a b : ℕ⦄ (hab : a < b) : r (f a) (f b) :=
  Nat.rel_of_forall_rel_succ_of_le_of_lt r (fun n _ ↦ h n) le_rfl hab


theorem Nat.rel_of_forall_rel_succ_of_le (r : β → β → Prop) [IsRefl β r] [IsTrans β r] {f : ℕ → β}
    (h : ∀ n, r (f n) (f (n + 1))) ⦃a b : ℕ⦄ (hab : a ≤ b) : r (f a) (f b) :=
  Nat.rel_of_forall_rel_succ_of_le_of_le r (fun n _ ↦ h n) le_rfl hab


theorem monotone_nat_of_le_succ {f : ℕ → α} (hf : ∀ n, f n ≤ f (n + 1)) : Monotone f :=
  Nat.rel_of_forall_rel_succ_of_le (· ≤ ·) hf


theorem antitone_nat_of_succ_le {f : ℕ → α} (hf : ∀ n, f (n + 1) ≤ f n) : Antitone f :=
  @monotone_nat_of_le_succ αᵒᵈ _ _ hf


theorem strictMono_nat_of_lt_succ {f : ℕ → α} (hf : ∀ n, f n < f (n + 1)) : StrictMono f :=
  Nat.rel_of_forall_rel_succ_of_lt (· < ·) hf


theorem strictAnti_nat_of_succ_lt {f : ℕ → α} (hf : ∀ n, f (n + 1) < f n) : StrictAnti f :=
  @strictMono_nat_of_lt_succ αᵒᵈ _ f hf


/-- If `α` is a preorder with no maximal elements, then there exists a strictly monotone function
`ℕ → α` with any prescribed value of `f 0`. -/
theorem exists_strictMono' [NoMaxOrder α] (a : α) : ∃ f : ℕ → α, StrictMono f ∧ f 0 = a := by
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    ⊢ Exists fun f => And (StrictMono f) (Eq (f 0) a)
  -/
  choose g hg using fun x : α ↦ exists_gt x
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : NoMaxOrder α
    a : α
    g : α → α
    hg : ∀ (x : α), LT.lt x (g x)
    ⊢ Exists fun f => And (StrictMono f) (Eq (f 0) a)
  -/
  exact ⟨fun n ↦ Nat.recOn n a fun _ ↦ g, strictMono_nat_of_lt_succ fun n ↦ hg _, rfl⟩
  /-
    🎉 no goals
  -/


/-- If `α` is a preorder with no maximal elements, then there exists a strictly antitone function
`ℕ → α` with any prescribed value of `f 0`. -/
theorem exists_strictAnti' [NoMinOrder α] (a : α) : ∃ f : ℕ → α, StrictAnti f ∧ f 0 = a :=
  exists_strictMono' (OrderDual.toDual a)


/-- If `α` is a nonempty preorder with no maximal elements, then there exists a strictly monotone
function `ℕ → α`. -/
theorem exists_strictMono [Nonempty α] [NoMaxOrder α] : ∃ f : ℕ → α, StrictMono f :=
  let ⟨a⟩ := ‹Nonempty α›
  let ⟨f, hf, _⟩ := exists_strictMono' a
  ⟨f, hf⟩


/-- If `α` is a nonempty preorder with no minimal elements, then there exists a strictly antitone
function `ℕ → α`. -/
theorem exists_strictAnti [Nonempty α] [NoMinOrder α] : ∃ f : ℕ → α, StrictAnti f :=
  exists_strictMono αᵒᵈ


lemma pow_self_mono : Monotone fun n : ℕ ↦ n ^ n := by
  /-
    ⊢ Monotone fun n => HPow.hPow n n
  -/
  refine monotone_nat_of_le_succ fun n ↦ ?_
  /-
    n : Nat
    ⊢ LE.le (HPow.hPow n n) (HPow.hPow (HAdd.hAdd n 1) (HAdd.hAdd n 1))
  -/
  rw [Nat.pow_succ]
  /-
    n : Nat
    ⊢ LE.le (HPow.hPow n n) (HMul.hMul (HPow.hPow (HAdd.hAdd n 1) n) (HAdd.hAdd n  …
  -/
  exact (Nat.pow_le_pow_left n.le_succ _).trans (Nat.le_mul_of_pos_right _ n.succ_pos)
  /-
    🎉 no goals
  -/


lemma pow_monotoneOn : MonotoneOn (fun p : ℕ × ℕ ↦ p.1 ^ p.2) {p | p.1 ≠ 0} := fun _p _ _q hq hpq ↦
  (Nat.pow_le_pow_left hpq.1 _).trans (Nat.pow_le_pow_right (Nat.pos_iff_ne_zero.2 hq) hpq.2)


lemma pow_self_strictMonoOn : StrictMonoOn (fun n : ℕ ↦ n ^ n) {n : ℕ | n ≠ 0} :=
  fun _m hm _n hn hmn ↦
    (Nat.pow_lt_pow_left hmn hm).trans_le (Nat.pow_le_pow_right (Nat.pos_iff_ne_zero.2 hn) hmn.le)


theorem Int.rel_of_forall_rel_succ_of_lt (r : β → β → Prop) [IsTrans β r] {f : ℤ → β}
    (h : ∀ n, r (f n) (f (n + 1))) ⦃a b : ℤ⦄ (hab : a < b) : r (f a) (f b) := by
  /-
    β : Type v
    r : β → β → Prop
    inst✝ : IsTrans β r
    f : Int → β
    h : ∀ (n : Int), r (f n) (f (HAdd.hAdd n 1))
    a b : Int
    hab : LT.lt a b
    ⊢ r (f a) (f b)
  -/
  rcases lt.dest hab with ⟨n, rfl⟩
  /-
    case intro
    β : Type v
    r : β → β → Prop
    inst✝ : IsTrans β r
    f : Int → β
    h : ∀ (n : Int), r (f n) (f (HAdd.hAdd n 1))
    a : Int
    n : Nat
    hab : LT.lt a (HAdd.hAdd a ↑n.succ)
    ⊢ r (f a) (f (HAdd.hAdd a ↑n.succ))
  -/
  clear hab
  induction n with
  | zero => rw [Int.ofNat_one]; apply h
  | succ n ihn => rw [Int.ofNat_succ, ← Int.add_assoc]; exact _root_.trans ihn (h _)


theorem Int.rel_of_forall_rel_succ_of_le (r : β → β → Prop) [IsRefl β r] [IsTrans β r] {f : ℤ → β}
    (h : ∀ n, r (f n) (f (n + 1))) ⦃a b : ℤ⦄ (hab : a ≤ b) : r (f a) (f b) :=
  hab.eq_or_lt.elim (fun h ↦ h ▸ refl _) fun h' ↦ Int.rel_of_forall_rel_succ_of_lt r h h'


theorem monotone_int_of_le_succ {f : ℤ → α} (hf : ∀ n, f n ≤ f (n + 1)) : Monotone f :=
  Int.rel_of_forall_rel_succ_of_le (· ≤ ·) hf


theorem antitone_int_of_succ_le {f : ℤ → α} (hf : ∀ n, f (n + 1) ≤ f n) : Antitone f :=
  Int.rel_of_forall_rel_succ_of_le (· ≥ ·) hf


theorem strictMono_int_of_lt_succ {f : ℤ → α} (hf : ∀ n, f n < f (n + 1)) : StrictMono f :=
  Int.rel_of_forall_rel_succ_of_lt (· < ·) hf


theorem strictAnti_int_of_succ_lt {f : ℤ → α} (hf : ∀ n, f (n + 1) < f n) : StrictAnti f :=
  Int.rel_of_forall_rel_succ_of_lt (· > ·) hf


/-- If `α` is a nonempty preorder with no minimal or maximal elements, then there exists a strictly
monotone function `f : ℤ → α`. -/
theorem exists_strictMono : ∃ f : ℤ → α, StrictMono f := by
  /-
    α : Type u
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    ⊢ Exists fun f => StrictMono f
  -/
  inhabit α
  /-
    α : Type u
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    inhabited_h : Inhabited α
    ⊢ Exists fun f => StrictMono f
  -/
  rcases Nat.exists_strictMono' (default : α) with ⟨f, hf, hf₀⟩
  /-
    case intro.intro
    α : Type u
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    inhabited_h : Inhabited α
    f : Nat → α
    hf : StrictMono f
    hf₀ : Eq (f 0) Inhabited.default
    ⊢ Exists fun f => StrictMono f
  -/
  rcases Nat.exists_strictAnti' (default : α) with ⟨g, hg, hg₀⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    inhabited_h : Inhabited α
    f : Nat → α
    hf : StrictMono f
    hf₀ : Eq (f 0) Inhabited.default
    g : Nat → α
    hg : StrictAnti g
    hg₀ : Eq (g 0) Inhabited.default
    ⊢ Exists fun f => StrictMono f
  -/
  refine ⟨fun n ↦ Int.casesOn n f fun n ↦ g (n + 1), strictMono_int_of_lt_succ ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝³ : Preorder α
    inst✝² : Nonempty α
    inst✝¹ : NoMinOrder α
    inst✝ : NoMaxOrder α
    inhabited_h : Inhabited α
    f : Nat → α
    hf : StrictMono f
    hf₀ : Eq (f 0) Inhabited.default
    g : Nat → α
    hg : StrictAnti g
    hg₀ : Eq (g 0) Inhabited.default
    ⊢ ∀ (n : Int), LT.lt (Int.casesOn n f fun n => g (HAdd.hAdd n 1)) (Int.casesOn …
  -/
  rintro (n | _ | n)
    /-
      case intro.intro.intro.intro.ofNat
      α : Type u
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : NoMinOrder α
      inst✝ : NoMaxOrder α
      inhabited_h : Inhabited α
      f : Nat → α
      hf : StrictMono f
      hf₀ : Eq (f 0) Inhabited.default
      g : Nat → α
      hg : StrictAnti g
      hg₀ : Eq (g 0) Inhabited.default
      n : Nat
      ⊢ LT.lt (Int.casesOn (Int.ofNat n) f fun n => g (HAdd.hAdd n 1)) (Int.casesOn  …
    -/
  · exact hf n.lt_succ_self
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.negSucc.zero
      α : Type u
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : NoMinOrder α
      inst✝ : NoMaxOrder α
      inhabited_h : Inhabited α
      f : Nat → α
      hf : StrictMono f
      hf₀ : Eq (f 0) Inhabited.default
      g : Nat → α
      hg : StrictAnti g
      hg₀ : Eq (g 0) Inhabited.default
      ⊢ LT.lt (Int.casesOn (Int.negSucc 0) f fun n => g (HAdd.hAdd n 1)) (Int.casesO …
    -/
  · show g 1 < f 0
    /-
      case intro.intro.intro.intro.negSucc.zero
      α : Type u
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : NoMinOrder α
      inst✝ : NoMaxOrder α
      inhabited_h : Inhabited α
      f : Nat → α
      hf : StrictMono f
      hf₀ : Eq (f 0) Inhabited.default
      g : Nat → α
      hg : StrictAnti g
      hg₀ : Eq (g 0) Inhabited.default
      ⊢ LT.lt (g 1) (f 0)
    -/
    rw [hf₀, ← hg₀]
    /-
      case intro.intro.intro.intro.negSucc.zero
      α : Type u
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : NoMinOrder α
      inst✝ : NoMaxOrder α
      inhabited_h : Inhabited α
      f : Nat → α
      hf : StrictMono f
      hf₀ : Eq (f 0) Inhabited.default
      g : Nat → α
      hg : StrictAnti g
      hg₀ : Eq (g 0) Inhabited.default
      ⊢ LT.lt (g 1) (g 0)
    -/
    exact hg Nat.zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.negSucc.succ
      α : Type u
      inst✝³ : Preorder α
      inst✝² : Nonempty α
      inst✝¹ : NoMinOrder α
      inst✝ : NoMaxOrder α
      inhabited_h : Inhabited α
      f : Nat → α
      hf : StrictMono f
      hf₀ : Eq (f 0) Inhabited.default
      g : Nat → α
      hg : StrictAnti g
      hg₀ : Eq (g 0) Inhabited.default
      n : Nat
      ⊢ LT.lt (Int.casesOn (Int.negSucc (HAdd.hAdd n 1)) f fun n => g (HAdd.hAdd n 1 …
    -/
  · exact hg (Nat.lt_succ_self _)
    /-
      🎉 no goals
    -/


/-- If `α` is a nonempty preorder with no minimal or maximal elements, then there exists a strictly
antitone function `f : ℤ → α`. -/
theorem exists_strictAnti : ∃ f : ℤ → α, StrictAnti f :=
  exists_strictMono αᵒᵈ


/-- If `f` is a monotone function from `ℕ` to a preorder such that `x` lies between `f n` and
  `f (n + 1)`, then `x` doesn't lie in the range of `f`. -/
theorem Monotone.ne_of_lt_of_lt_nat {f : ℕ → α} (hf : Monotone f) (n : ℕ) {x : α} (h1 : f n < x)
    (h2 : x < f (n + 1)) (a : ℕ) : f a ≠ x := by
  /-
    α : Type u
    inst✝ : Preorder α
    f : Nat → α
    hf : Monotone f
    n : Nat
    x : α
    h1 : LT.lt (f n) x
    h2 : LT.lt x (f (HAdd.hAdd n 1))
    a : Nat
    ⊢ Ne (f a) x
  -/
  rintro rfl
  /-
    α : Type u
    inst✝ : Preorder α
    f : Nat → α
    hf : Monotone f
    n a : Nat
    h1 : LT.lt (f n) (f a)
    h2 : LT.lt (f a) (f (HAdd.hAdd n 1))
    ⊢ False
  -/
  exact (hf.reflect_lt h1).not_le (Nat.le_of_lt_succ <| hf.reflect_lt h2)
  /-
    🎉 no goals
  -/


/-- If `f` is an antitone function from `ℕ` to a preorder such that `x` lies between `f (n + 1)` and
`f n`, then `x` doesn't lie in the range of `f`. -/
theorem Antitone.ne_of_lt_of_lt_nat {f : ℕ → α} (hf : Antitone f) (n : ℕ) {x : α}
    (h1 : f (n + 1) < x) (h2 : x < f n) (a : ℕ) : f a ≠ x := by
  /-
    α : Type u
    inst✝ : Preorder α
    f : Nat → α
    hf : Antitone f
    n : Nat
    x : α
    h1 : LT.lt (f (HAdd.hAdd n 1)) x
    h2 : LT.lt x (f n)
    a : Nat
    ⊢ Ne (f a) x
  -/
  rintro rfl
  /-
    α : Type u
    inst✝ : Preorder α
    f : Nat → α
    hf : Antitone f
    n a : Nat
    h1 : LT.lt (f (HAdd.hAdd n 1)) (f a)
    h2 : LT.lt (f a) (f n)
    ⊢ False
  -/
  exact (hf.reflect_lt h2).not_le (Nat.le_of_lt_succ <| hf.reflect_lt h1)
  /-
    🎉 no goals
  -/


/-- If `f` is a monotone function from `ℤ` to a preorder and `x` lies between `f n` and
  `f (n + 1)`, then `x` doesn't lie in the range of `f`. -/
theorem Monotone.ne_of_lt_of_lt_int {f : ℤ → α} (hf : Monotone f) (n : ℤ) {x : α} (h1 : f n < x)
    (h2 : x < f (n + 1)) (a : ℤ) : f a ≠ x := by
  /-
    α : Type u
    inst✝ : Preorder α
    f : Int → α
    hf : Monotone f
    n : Int
    x : α
    h1 : LT.lt (f n) x
    h2 : LT.lt x (f (HAdd.hAdd n 1))
    a : Int
    ⊢ Ne (f a) x
  -/
  rintro rfl
  /-
    α : Type u
    inst✝ : Preorder α
    f : Int → α
    hf : Monotone f
    n a : Int
    h1 : LT.lt (f n) (f a)
    h2 : LT.lt (f a) (f (HAdd.hAdd n 1))
    ⊢ False
  -/
  exact (hf.reflect_lt h1).not_le (Int.le_of_lt_add_one <| hf.reflect_lt h2)
  /-
    🎉 no goals
  -/


/-- If `f` is an antitone function from `ℤ` to a preorder and `x` lies between `f (n + 1)` and
`f n`, then `x` doesn't lie in the range of `f`. -/
theorem Antitone.ne_of_lt_of_lt_int {f : ℤ → α} (hf : Antitone f) (n : ℤ) {x : α}
    (h1 : f (n + 1) < x) (h2 : x < f n) (a : ℤ) : f a ≠ x := by
  /-
    α : Type u
    inst✝ : Preorder α
    f : Int → α
    hf : Antitone f
    n : Int
    x : α
    h1 : LT.lt (f (HAdd.hAdd n 1)) x
    h2 : LT.lt x (f n)
    a : Int
    ⊢ Ne (f a) x
  -/
  rintro rfl
  /-
    α : Type u
    inst✝ : Preorder α
    f : Int → α
    hf : Antitone f
    n a : Int
    h1 : LT.lt (f (HAdd.hAdd n 1)) (f a)
    h2 : LT.lt (f a) (f n)
    ⊢ False
  -/
  exact (hf.reflect_lt h2).not_le (Int.le_of_lt_add_one <| hf.reflect_lt h1)
  /-
    🎉 no goals
  -/


theorem Subtype.mono_coe [Preorder α] (t : Set α) : Monotone ((↑) : Subtype t → α) :=
  fun _ _ ↦ id


theorem Subtype.strictMono_coe [Preorder α] (t : Set α) :
    StrictMono ((↑) : Subtype t → α) :=
  fun _ _ ↦ id


theorem monotone_fst : Monotone (@Prod.fst α β) := fun _ _ ↦ And.left


theorem monotone_snd : Monotone (@Prod.snd α β) := fun _ _ ↦ And.right


theorem Monotone.prod_map (hf : Monotone f) (hg : Monotone g) : Monotone (Prod.map f g) :=
  fun _ _ h ↦ ⟨hf h.1, hg h.2⟩


theorem Antitone.prod_map (hf : Antitone f) (hg : Antitone g) : Antitone (Prod.map f g) :=
  fun _ _ h ↦ ⟨hf h.1, hg h.2⟩


theorem StrictMono.prod_map (hf : StrictMono f) (hg : StrictMono g) : StrictMono (Prod.map f g) :=
  fun a b ↦ by
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : Preorder γ
    inst✝ : Preorder δ
    f : α → γ
    g : β → δ
    hf : StrictMono f
    hg : StrictMono g
    a b : Prod α β
    ⊢ LT.lt a b → LT.lt (Prod.map f g a) (Prod.map f g b)
  -/
  simp only [Prod.lt_iff]
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : Preorder γ
    inst✝ : Preorder δ
    f : α → γ
    g : β → δ
    hf : StrictMono f
    hg : StrictMono g
    a b : Prod α β
    ⊢ Or (And (LT.lt a.fst b.fst) (LE.le a.snd b.snd)) (And (LE.le a.fst b.fst) (L …
  -/
  exact Or.imp (And.imp hf.imp hg.monotone.imp) (And.imp hf.monotone.imp hg.imp)
  /-
    🎉 no goals
  -/


theorem StrictAnti.prod_map (hf : StrictAnti f) (hg : StrictAnti g) : StrictAnti (Prod.map f g) :=
  fun a b ↦ by
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : Preorder γ
    inst✝ : Preorder δ
    f : α → γ
    g : β → δ
    hf : StrictAnti f
    hg : StrictAnti g
    a b : Prod α β
    ⊢ LT.lt a b → LT.lt (Prod.map f g b) (Prod.map f g a)
  -/
  simp only [Prod.lt_iff]
  /-
    α : Type u
    β : Type v
    γ : Type w
    δ : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : Preorder γ
    inst✝ : Preorder δ
    f : α → γ
    g : β → δ
    hf : StrictAnti f
    hg : StrictAnti g
    a b : Prod α β
    ⊢ Or (And (LT.lt a.fst b.fst) (LE.le a.snd b.snd)) (And (LE.le a.fst b.fst) (L …
  -/
  exact Or.imp (And.imp hf.imp hg.antitone.imp) (And.imp hf.antitone.imp hg.imp)
  /-
    🎉 no goals
  -/


theorem update_mono : Monotone (update f i) := fun _ _ => update_le_update_iff'.2


theorem update_strictMono : StrictMono (update f i) := fun _ _ => update_lt_update_iff.2


theorem const_mono : Monotone (const β : α → β → α) := fun _ _ h _ ↦ h


theorem const_strictMono [Nonempty β] : StrictMono (const β : α → β → α) :=
  fun _ _ ↦ const_lt_const.2


lemma monotone_iff_apply₂ : Monotone f ↔ ∀ i, Monotone (f · i) := by
  /-
    ι : Type u_1
    α : Type u
    β : ι → Type u_4
    inst✝¹ : (i : ι) → Preorder (β i)
    inst✝ : Preorder α
    f : α → (i : ι) → β i
    ⊢ Iff (Monotone f) (∀ (i : ι), Monotone fun x => f x i)
  -/
  simp [Monotone, Pi.le_def, @forall_swap ι]
  /-
    🎉 no goals
  -/


lemma antitone_iff_apply₂ : Antitone f ↔ ∀ i, Antitone (f · i) := by
  /-
    ι : Type u_1
    α : Type u
    β : ι → Type u_4
    inst✝¹ : (i : ι) → Preorder (β i)
    inst✝ : Preorder α
    f : α → (i : ι) → β i
    ⊢ Iff (Antitone f) (∀ (i : ι), Antitone fun x => f x i)
  -/
  simp [Antitone, Pi.le_def, @forall_swap ι]
  /-
    🎉 no goals
  -/


alias ⟨Monotone.apply₂, Monotone.of_apply₂⟩ := monotone_iff_apply₂

alias ⟨Antitone.apply₂, Antitone.of_apply₂⟩ := antitone_iff_apply₂


/-- A monotone function `f : ℕ → ℕ` bounded by `b`, which is constant after stabilising for the
first time, stabilises in at most `b` steps. -/
lemma Nat.stabilises_of_monotone {f : ℕ → ℕ} {b n : ℕ} (hfmono : Monotone f) (hfb : ∀ m, f m ≤ b)
    (hfstab : ∀ m, f m = f (m + 1) → f (m + 1) = f (m + 2)) (hbn : b ≤ n) : f n = f b := by
  obtain ⟨m, hmb, hm⟩ : ∃ m ≤ b, f m = f (m + 1) := by
    contrapose! hfb
    let rec strictMono : ∀ m ≤ b + 1, m ≤ f m
    | 0, _ => Nat.zero_le _
    | m + 1, hmb => (strictMono _ <| m.le_succ.trans hmb).trans_lt <| (hfmono m.le_succ).lt_of_ne <|
        hfb _ <| Nat.le_of_succ_le_succ hmb
    exact ⟨b + 1, strictMono _ le_rfl⟩
  replace key : ∀ k : ℕ, f (m + k) = f (m + k + 1) ∧ f (m + k) = f m := fun k =>
    Nat.rec ⟨hm, rfl⟩ (fun k ih => ⟨hfstab _ ih.1, ih.1.symm.trans ih.2⟩) k
  replace key : ∀ k ≥ m, f k = f m := fun k hk =>
    (congr_arg f (Nat.add_sub_of_le hk)).symm.trans (key (k - m)).2
  /-
    case intro.intro
    f : Nat → Nat
    b n : Nat
    hfmono : Monotone f
    hfb : ∀ (m : Nat), LE.le (f m) b
    hfstab : ∀ (m : Nat), Eq (f m) (f (HAdd.hAdd m 1)) → Eq (f (HAdd.hAdd m 1)) (f …
    hbn : LE.le b n
    m : Nat
    hmb : LE.le m b
    hm : Eq (f m) (f (HAdd.hAdd m 1))
    key : ∀ (k : Nat), GE.ge k m → Eq (f k) (f m)
    ⊢ Eq (f n) (f b)
  -/
  exact (key n (hmb.trans hbn)).trans (key b hmb).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")]
alias Group.card_pow_eq_card_pow_card_univ_aux := Nat.stabilises_of_monotone


@[deprecated (since := "2024-11-27")]
alias Group.card_nsmul_eq_card_nsmulpow_card_univ_aux := Nat.stabilises_of_monotone

