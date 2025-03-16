/-- A function `f` between preorders is left order continuous if it preserves all suprema.  We
define it using `IsLUB` instead of `sSup` so that the proof works both for complete lattices and
conditionally complete lattices. -/
def LeftOrdContinuous [Preorder α] [Preorder β] (f : α → β) :=
  ∀ ⦃s : Set α⦄ ⦃x⦄, IsLUB s x → IsLUB (f '' s) (f x)


/-- A function `f` between preorders is right order continuous if it preserves all infima.  We
define it using `IsGLB` instead of `sInf` so that the proof works both for complete lattices and
conditionally complete lattices. -/
def RightOrdContinuous [Preorder α] [Preorder β] (f : α → β) :=
  ∀ ⦃s : Set α⦄ ⦃x⦄, IsGLB s x → IsGLB (f '' s) (f x)


protected theorem id : LeftOrdContinuous (id : α → α) := fun s x h => by
  /-
    α : Type u
    inst✝ : Preorder α
    s : Set α
    x : α
    h : IsLUB s x
    ⊢ IsLUB (Set.image id s) (id x)
  -/
  simpa only [image_id] using h
  /-
    🎉 no goals
  -/


protected theorem order_dual : LeftOrdContinuous f → RightOrdContinuous (toDual ∘ f ∘ ofDual) :=
  id


theorem map_isGreatest (hf : LeftOrdContinuous f) {s : Set α} {x : α} (h : IsGreatest s x) :
    IsGreatest (f '' s) (f x) :=
  ⟨mem_image_of_mem f h.1, (hf h.isLUB).1⟩


theorem mono (hf : LeftOrdContinuous f) : Monotone f := fun a₁ a₂ h =>
                                                   /-
                                                     α : Type u
                                                     β : Type v
                                                     inst✝¹ : Preorder α
                                                     inst✝ : Preorder β
                                                     f : α → β
                                                     hf : LeftOrdContinuous f
                                                     a₁ a₂ : α
                                                     h : LE.le a₁ a₂
                                                     ⊢ Membership.mem (upperBounds (Insert.insert a₁ (Singleton.singleton a₂))) a₂
                                                   -/
  have : IsGreatest {a₁, a₂} a₂ := ⟨Or.inr rfl, by simp [*]⟩
                                                   /-
                                                     🎉 no goals
                                                   -/
  (hf.map_isGreatest this).2 <| mem_image_of_mem _ (Or.inl rfl)


theorem comp (hg : LeftOrdContinuous g) (hf : LeftOrdContinuous f) : LeftOrdContinuous (g ∘ f) :=
                  /-
                    α : Type u
                    β : Type v
                    γ : Type w
                    inst✝² : Preorder α
                    inst✝¹ : Preorder β
                    inst✝ : Preorder γ
                    g : β → γ
                    f : α → β
                    hg : LeftOrdContinuous g
                    hf : LeftOrdContinuous f
                    s : Set α
                    x : α
                    h : IsLUB s x
                    ⊢ IsLUB (Set.image (Function.comp g f) s) (Function.comp g f x)
                  -/
  fun s x h => by simpa only [image_image] using hg (hf h)
                  /-
                    🎉 no goals
                  -/

-- Porting note: how to do this in non-tactic mode?

protected theorem iterate {f : α → α} (hf : LeftOrdContinuous f) (n : ℕ) :
    LeftOrdContinuous f^[n] := by
  induction n with
  | zero => exact LeftOrdContinuous.id α
  | succ n ihn => exact ihn.comp hf


theorem map_sup (hf : LeftOrdContinuous f) (x y : α) : f (x ⊔ y) = f x ⊔ f y :=
                               /-
                                 α : Type u
                                 β : Type v
                                 inst✝¹ : SemilatticeSup α
                                 inst✝ : SemilatticeSup β
                                 f : α → β
                                 hf : LeftOrdContinuous f
                                 x y : α
                                 ⊢ IsLUB (Set.image f (Insert.insert x (Singleton.singleton y))) (Max.max (f x) …
                               -/
  (hf isLUB_pair).unique <| by simp only [image_pair, isLUB_pair]
                               /-
                                 🎉 no goals
                               -/


theorem le_iff (hf : LeftOrdContinuous f) (h : Injective f) {x y} : f x ≤ f y ↔ x ≤ y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : SemilatticeSup α
    inst✝ : SemilatticeSup β
    f : α → β
    hf : LeftOrdContinuous f
    h : Function.Injective f
    x y : α
    ⊢ Iff (LE.le (f x) (f y)) (LE.le x y)
  -/
  simp only [← sup_eq_right, ← hf.map_sup, h.eq_iff]
  /-
    🎉 no goals
  -/


theorem lt_iff (hf : LeftOrdContinuous f) (h : Injective f) {x y} : f x < f y ↔ x < y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : SemilatticeSup α
    inst✝ : SemilatticeSup β
    f : α → β
    hf : LeftOrdContinuous f
    h : Function.Injective f
    x y : α
    ⊢ Iff (LT.lt (f x) (f y)) (LT.lt x y)
  -/
  simp only [lt_iff_le_not_le, hf.le_iff h]
  /-
    🎉 no goals
  -/


/-- Convert an injective left order continuous function to an order embedding. -/
def toOrderEmbedding (hf : LeftOrdContinuous f) (h : Injective f) : α ↪o β :=
  ⟨⟨f, h⟩, hf.le_iff h⟩


@[simp]
theorem coe_toOrderEmbedding (hf : LeftOrdContinuous f) (h : Injective f) :
    ⇑(hf.toOrderEmbedding f h) = f :=
  rfl


theorem map_sSup' (hf : LeftOrdContinuous f) (s : Set α) : f (sSup s) = sSup (f '' s) :=
  (hf <| isLUB_sSup s).sSup_eq.symm


theorem map_sSup (hf : LeftOrdContinuous f) (s : Set α) : f (sSup s) = ⨆ x ∈ s, f x := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : α → β
    hf : LeftOrdContinuous f
    s : Set α
    ⊢ Eq (f (SupSet.sSup s)) (iSup fun x => iSup fun h => f x)
  -/
  rw [hf.map_sSup', sSup_image]
  /-
    🎉 no goals
  -/


theorem map_iSup (hf : LeftOrdContinuous f) (g : ι → α) : f (⨆ i, g i) = ⨆ i, f (g i) := by
  /-
    α : Type u
    β : Type v
    ι : Sort x
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : α → β
    hf : LeftOrdContinuous f
    g : ι → α
    ⊢ Eq (f (iSup fun i => g i)) (iSup fun i => f (g i))
  -/
  simp only [iSup, hf.map_sSup', ← range_comp]
  /-
    α : Type u
    β : Type v
    ι : Sort x
    inst✝¹ : CompleteLattice α
    inst✝ : CompleteLattice β
    f : α → β
    hf : LeftOrdContinuous f
    g : ι → α
    ⊢ Eq (SupSet.sSup (Set.range (Function.comp f fun i => g i))) (SupSet.sSup (Se …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_csSup (hf : LeftOrdContinuous f) {s : Set α} (sne : s.Nonempty) (sbdd : BddAbove s) :
    f (sSup s) = sSup (f '' s) :=
  ((hf <| isLUB_csSup sne sbdd).csSup_eq <| sne.image f).symm


theorem map_ciSup (hf : LeftOrdContinuous f) {g : ι → α} (hg : BddAbove (range g)) :
    f (⨆ i, g i) = ⨆ i, f (g i) := by
  /-
    α : Type u
    β : Type v
    ι : Sort x
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : Nonempty ι
    f : α → β
    hf : LeftOrdContinuous f
    g : ι → α
    hg : BddAbove (Set.range g)
    ⊢ Eq (f (iSup fun i => g i)) (iSup fun i => f (g i))
  -/
  simp only [iSup, hf.map_csSup (range_nonempty _) hg, ← range_comp]
  /-
    α : Type u
    β : Type v
    ι : Sort x
    inst✝² : ConditionallyCompleteLattice α
    inst✝¹ : ConditionallyCompleteLattice β
    inst✝ : Nonempty ι
    f : α → β
    hf : LeftOrdContinuous f
    g : ι → α
    hg : BddAbove (Set.range g)
    ⊢ Eq (SupSet.sSup (Set.range (Function.comp f g))) (SupSet.sSup (Set.range fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem id : RightOrdContinuous (id : α → α) := fun s x h => by
  /-
    α : Type u
    inst✝ : Preorder α
    s : Set α
    x : α
    h : IsGLB s x
    ⊢ IsGLB (Set.image id s) (id x)
  -/
  simpa only [image_id] using h
  /-
    🎉 no goals
  -/


protected theorem orderDual : RightOrdContinuous f → LeftOrdContinuous (toDual ∘ f ∘ ofDual) :=
  id


theorem map_isLeast (hf : RightOrdContinuous f) {s : Set α} {x : α} (h : IsLeast s x) :
    IsLeast (f '' s) (f x) :=
  hf.orderDual.map_isGreatest h


theorem mono (hf : RightOrdContinuous f) : Monotone f :=
  hf.orderDual.mono.dual


theorem comp (hg : RightOrdContinuous g) (hf : RightOrdContinuous f) : RightOrdContinuous (g ∘ f) :=
  hg.orderDual.comp hf.orderDual


protected theorem iterate {f : α → α} (hf : RightOrdContinuous f) (n : ℕ) :
    RightOrdContinuous f^[n] :=
  hf.orderDual.iterate n


theorem map_inf (hf : RightOrdContinuous f) (x y : α) : f (x ⊓ y) = f x ⊓ f y :=
  hf.orderDual.map_sup x y


theorem le_iff (hf : RightOrdContinuous f) (h : Injective f) {x y} : f x ≤ f y ↔ x ≤ y :=
  hf.orderDual.le_iff h


theorem lt_iff (hf : RightOrdContinuous f) (h : Injective f) {x y} : f x < f y ↔ x < y :=
  hf.orderDual.lt_iff h


/-- Convert an injective left order continuous function to an `OrderEmbedding`. -/
def toOrderEmbedding (hf : RightOrdContinuous f) (h : Injective f) : α ↪o β :=
  ⟨⟨f, h⟩, hf.le_iff h⟩


@[simp]
theorem coe_toOrderEmbedding (hf : RightOrdContinuous f) (h : Injective f) :
    ⇑(hf.toOrderEmbedding f h) = f :=
  rfl


theorem map_sInf' (hf : RightOrdContinuous f) (s : Set α) : f (sInf s) = sInf (f '' s) :=
  hf.orderDual.map_sSup' s


theorem map_sInf (hf : RightOrdContinuous f) (s : Set α) : f (sInf s) = ⨅ x ∈ s, f x :=
  hf.orderDual.map_sSup s


theorem map_iInf (hf : RightOrdContinuous f) (g : ι → α) : f (⨅ i, g i) = ⨅ i, f (g i) :=
  hf.orderDual.map_iSup g


theorem map_csInf (hf : RightOrdContinuous f) {s : Set α} (sne : s.Nonempty) (sbdd : BddBelow s) :
    f (sInf s) = sInf (f '' s) :=
  hf.orderDual.map_csSup sne sbdd


theorem map_ciInf (hf : RightOrdContinuous f) {g : ι → α} (hg : BddBelow (range g)) :
    f (⨅ i, g i) = ⨅ i, f (g i) :=
  hf.orderDual.map_ciSup hg


protected theorem leftOrdContinuous : LeftOrdContinuous e := fun _ _ hx =>
  ⟨Monotone.mem_upperBounds_image (fun _ _ => e.map_rel_iff.2) hx.1, fun _ hy =>
    e.rel_symm_apply.1 <|
      (isLUB_le_iff hx).2 fun _ hx' => e.rel_symm_apply.2 <| hy <| mem_image_of_mem _ hx'⟩


protected theorem rightOrdContinuous : RightOrdContinuous e :=
  OrderIso.leftOrdContinuous e.dual


