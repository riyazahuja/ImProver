/-- `IsMinFilter f l a` means that `f a ≤ f x` for all `x` in some `l`-neighborhood of `a` -/
def IsMinFilter : Prop :=
  ∀ᶠ x in l, f a ≤ f x


/-- `is_maxFilter f l a` means that `f x ≤ f a` for all `x` in some `l`-neighborhood of `a` -/
def IsMaxFilter : Prop :=
  ∀ᶠ x in l, f x ≤ f a


/-- `IsExtrFilter f l a` means `IsMinFilter f l a` or `IsMaxFilter f l a` -/
def IsExtrFilter : Prop :=
  IsMinFilter f l a ∨ IsMaxFilter f l a


/-- `IsMinOn f s a` means that `f a ≤ f x` for all `x ∈ s`. Note that we do not assume `a ∈ s`. -/
def IsMinOn :=
  IsMinFilter f (𝓟 s) a


/-- `IsMaxOn f s a` means that `f x ≤ f a` for all `x ∈ s`. Note that we do not assume `a ∈ s`. -/
def IsMaxOn :=
  IsMaxFilter f (𝓟 s) a


/-- `IsExtrOn f s a` means `IsMinOn f s a` or `IsMaxOn f s a` -/
def IsExtrOn : Prop :=
  IsExtrFilter f (𝓟 s) a


theorem IsExtrOn.elim {p : Prop} : IsExtrOn f s a → (IsMinOn f s a → p) → (IsMaxOn f s a → p) → p :=
  Or.elim


theorem isMinOn_iff : IsMinOn f s a ↔ ∀ x ∈ s, f a ≤ f x :=
  Iff.rfl


theorem isMaxOn_iff : IsMaxOn f s a ↔ ∀ x ∈ s, f x ≤ f a :=
  Iff.rfl


theorem isMinOn_univ_iff : IsMinOn f univ a ↔ ∀ x, f a ≤ f x :=
  univ_subset_iff.trans eq_univ_iff_forall


theorem isMaxOn_univ_iff : IsMaxOn f univ a ↔ ∀ x, f x ≤ f a :=
  univ_subset_iff.trans eq_univ_iff_forall


theorem IsMinFilter.tendsto_principal_Ici (h : IsMinFilter f l a) : Tendsto f l (𝓟 <| Ici (f a)) :=
  tendsto_principal.2 h


theorem IsMaxFilter.tendsto_principal_Iic (h : IsMaxFilter f l a) : Tendsto f l (𝓟 <| Iic (f a)) :=
  tendsto_principal.2 h


theorem IsMinFilter.isExtr : IsMinFilter f l a → IsExtrFilter f l a :=
  Or.inl


theorem IsMaxFilter.isExtr : IsMaxFilter f l a → IsExtrFilter f l a :=
  Or.inr


theorem IsMinOn.isExtr (h : IsMinOn f s a) : IsExtrOn f s a :=
  IsMinFilter.isExtr h


theorem IsMaxOn.isExtr (h : IsMaxOn f s a) : IsExtrOn f s a :=
  IsMaxFilter.isExtr h


theorem isMinFilter_const {b : β} : IsMinFilter (fun _ => b) l a :=
  univ_mem' fun _ => le_rfl


theorem isMaxFilter_const {b : β} : IsMaxFilter (fun _ => b) l a :=
  univ_mem' fun _ => le_rfl


theorem isExtrFilter_const {b : β} : IsExtrFilter (fun _ => b) l a :=
  isMinFilter_const.isExtr


theorem isMinOn_const {b : β} : IsMinOn (fun _ => b) s a :=
  isMinFilter_const


theorem isMaxOn_const {b : β} : IsMaxOn (fun _ => b) s a :=
  isMaxFilter_const


theorem isExtrOn_const {b : β} : IsExtrOn (fun _ => b) s a :=
  isExtrFilter_const


theorem isMinFilter_dual_iff : IsMinFilter (toDual ∘ f) l a ↔ IsMaxFilter f l a :=
  Iff.rfl


theorem isMaxFilter_dual_iff : IsMaxFilter (toDual ∘ f) l a ↔ IsMinFilter f l a :=
  Iff.rfl


theorem isExtrFilter_dual_iff : IsExtrFilter (toDual ∘ f) l a ↔ IsExtrFilter f l a :=
  or_comm


alias ⟨IsMinFilter.undual, IsMaxFilter.dual⟩ := isMinFilter_dual_iff


alias ⟨IsMaxFilter.undual, IsMinFilter.dual⟩ := isMaxFilter_dual_iff


alias ⟨IsExtrFilter.undual, IsExtrFilter.dual⟩ := isExtrFilter_dual_iff


theorem isMinOn_dual_iff : IsMinOn (toDual ∘ f) s a ↔ IsMaxOn f s a :=
  Iff.rfl


theorem isMaxOn_dual_iff : IsMaxOn (toDual ∘ f) s a ↔ IsMinOn f s a :=
  Iff.rfl


theorem isExtrOn_dual_iff : IsExtrOn (toDual ∘ f) s a ↔ IsExtrOn f s a :=
  or_comm


alias ⟨IsMinOn.undual, IsMaxOn.dual⟩ := isMinOn_dual_iff


alias ⟨IsMaxOn.undual, IsMinOn.dual⟩ := isMaxOn_dual_iff


alias ⟨IsExtrOn.undual, IsExtrOn.dual⟩ := isExtrOn_dual_iff


theorem IsMinFilter.filter_mono (h : IsMinFilter f l a) (hl : l' ≤ l) : IsMinFilter f l' a :=
  hl h


theorem IsMaxFilter.filter_mono (h : IsMaxFilter f l a) (hl : l' ≤ l) : IsMaxFilter f l' a :=
  hl h


theorem IsExtrFilter.filter_mono (h : IsExtrFilter f l a) (hl : l' ≤ l) : IsExtrFilter f l' a :=
  h.elim (fun h => (h.filter_mono hl).isExtr) fun h => (h.filter_mono hl).isExtr


theorem IsMinFilter.filter_inf (h : IsMinFilter f l a) (l') : IsMinFilter f (l ⊓ l') a :=
  h.filter_mono inf_le_left


theorem IsMaxFilter.filter_inf (h : IsMaxFilter f l a) (l') : IsMaxFilter f (l ⊓ l') a :=
  h.filter_mono inf_le_left


theorem IsExtrFilter.filter_inf (h : IsExtrFilter f l a) (l') : IsExtrFilter f (l ⊓ l') a :=
  h.filter_mono inf_le_left


theorem IsMinOn.on_subset (hf : IsMinOn f t a) (h : s ⊆ t) : IsMinOn f s a :=
  hf.filter_mono <| principal_mono.2 h


theorem IsMaxOn.on_subset (hf : IsMaxOn f t a) (h : s ⊆ t) : IsMaxOn f s a :=
  hf.filter_mono <| principal_mono.2 h


theorem IsExtrOn.on_subset (hf : IsExtrOn f t a) (h : s ⊆ t) : IsExtrOn f s a :=
  hf.filter_mono <| principal_mono.2 h


theorem IsMinOn.inter (hf : IsMinOn f s a) (t) : IsMinOn f (s ∩ t) a :=
  hf.on_subset inter_subset_left


theorem IsMaxOn.inter (hf : IsMaxOn f s a) (t) : IsMaxOn f (s ∩ t) a :=
  hf.on_subset inter_subset_left


theorem IsExtrOn.inter (hf : IsExtrOn f s a) (t) : IsExtrOn f (s ∩ t) a :=
  hf.on_subset inter_subset_left


theorem IsMinFilter.comp_mono (hf : IsMinFilter f l a) {g : β → γ} (hg : Monotone g) :
    IsMinFilter (g ∘ f) l a :=
  mem_of_superset hf fun _x hx => hg hx


theorem IsMaxFilter.comp_mono (hf : IsMaxFilter f l a) {g : β → γ} (hg : Monotone g) :
    IsMaxFilter (g ∘ f) l a :=
  mem_of_superset hf fun _x hx => hg hx


theorem IsExtrFilter.comp_mono (hf : IsExtrFilter f l a) {g : β → γ} (hg : Monotone g) :
    IsExtrFilter (g ∘ f) l a :=
  hf.elim (fun hf => (hf.comp_mono hg).isExtr) fun hf => (hf.comp_mono hg).isExtr


theorem IsMinFilter.comp_antitone (hf : IsMinFilter f l a) {g : β → γ} (hg : Antitone g) :
    IsMaxFilter (g ∘ f) l a :=
  hf.dual.comp_mono fun _ _ h => hg h


theorem IsMaxFilter.comp_antitone (hf : IsMaxFilter f l a) {g : β → γ} (hg : Antitone g) :
    IsMinFilter (g ∘ f) l a :=
  hf.dual.comp_mono fun _ _ h => hg h


theorem IsExtrFilter.comp_antitone (hf : IsExtrFilter f l a) {g : β → γ} (hg : Antitone g) :
    IsExtrFilter (g ∘ f) l a :=
  hf.dual.comp_mono fun _ _ h => hg h


theorem IsMinOn.comp_mono (hf : IsMinOn f s a) {g : β → γ} (hg : Monotone g) :
    IsMinOn (g ∘ f) s a :=
  IsMinFilter.comp_mono hf hg


theorem IsMaxOn.comp_mono (hf : IsMaxOn f s a) {g : β → γ} (hg : Monotone g) :
    IsMaxOn (g ∘ f) s a :=
  IsMaxFilter.comp_mono hf hg


theorem IsExtrOn.comp_mono (hf : IsExtrOn f s a) {g : β → γ} (hg : Monotone g) :
    IsExtrOn (g ∘ f) s a :=
  IsExtrFilter.comp_mono hf hg


theorem IsMinOn.comp_antitone (hf : IsMinOn f s a) {g : β → γ} (hg : Antitone g) :
    IsMaxOn (g ∘ f) s a :=
  IsMinFilter.comp_antitone hf hg


theorem IsMaxOn.comp_antitone (hf : IsMaxOn f s a) {g : β → γ} (hg : Antitone g) :
    IsMinOn (g ∘ f) s a :=
  IsMaxFilter.comp_antitone hf hg


theorem IsExtrOn.comp_antitone (hf : IsExtrOn f s a) {g : β → γ} (hg : Antitone g) :
    IsExtrOn (g ∘ f) s a :=
  IsExtrFilter.comp_antitone hf hg


theorem IsMinFilter.bicomp_mono [Preorder δ] {op : β → γ → δ}
    (hop : ((· ≤ ·) ⇒ (· ≤ ·) ⇒ (· ≤ ·)) op op) (hf : IsMinFilter f l a) {g : α → γ}
    (hg : IsMinFilter g l a) : IsMinFilter (fun x => op (f x) (g x)) l a :=
  mem_of_superset (inter_mem hf hg) fun _x ⟨hfx, hgx⟩ => hop hfx hgx


theorem IsMaxFilter.bicomp_mono [Preorder δ] {op : β → γ → δ}
    (hop : ((· ≤ ·) ⇒ (· ≤ ·) ⇒ (· ≤ ·)) op op) (hf : IsMaxFilter f l a) {g : α → γ}
    (hg : IsMaxFilter g l a) : IsMaxFilter (fun x => op (f x) (g x)) l a :=
  mem_of_superset (inter_mem hf hg) fun _x ⟨hfx, hgx⟩ => hop hfx hgx

-- No `Extr` version because we need `hf` and `hg` to be of the same kind

theorem IsMinOn.bicomp_mono [Preorder δ] {op : β → γ → δ}
    (hop : ((· ≤ ·) ⇒ (· ≤ ·) ⇒ (· ≤ ·)) op op) (hf : IsMinOn f s a) {g : α → γ}
    (hg : IsMinOn g s a) : IsMinOn (fun x => op (f x) (g x)) s a :=
  IsMinFilter.bicomp_mono hop hf hg


theorem IsMaxOn.bicomp_mono [Preorder δ] {op : β → γ → δ}
    (hop : ((· ≤ ·) ⇒ (· ≤ ·) ⇒ (· ≤ ·)) op op) (hf : IsMaxOn f s a) {g : α → γ}
    (hg : IsMaxOn g s a) : IsMaxOn (fun x => op (f x) (g x)) s a :=
  IsMaxFilter.bicomp_mono hop hf hg


theorem IsMinFilter.comp_tendsto {g : δ → α} {l' : Filter δ} {b : δ} (hf : IsMinFilter f l (g b))
    (hg : Tendsto g l' l) : IsMinFilter (f ∘ g) l' b :=
  hg hf


theorem IsMaxFilter.comp_tendsto {g : δ → α} {l' : Filter δ} {b : δ} (hf : IsMaxFilter f l (g b))
    (hg : Tendsto g l' l) : IsMaxFilter (f ∘ g) l' b :=
  hg hf


theorem IsExtrFilter.comp_tendsto {g : δ → α} {l' : Filter δ} {b : δ} (hf : IsExtrFilter f l (g b))
    (hg : Tendsto g l' l) : IsExtrFilter (f ∘ g) l' b :=
  hf.elim (fun hf => (hf.comp_tendsto hg).isExtr) fun hf => (hf.comp_tendsto hg).isExtr


theorem IsMinOn.on_preimage (g : δ → α) {b : δ} (hf : IsMinOn f s (g b)) :
    IsMinOn (f ∘ g) (g ⁻¹' s) b :=
  hf.comp_tendsto (tendsto_principal_principal.mpr <| Subset.refl _)


theorem IsMaxOn.on_preimage (g : δ → α) {b : δ} (hf : IsMaxOn f s (g b)) :
    IsMaxOn (f ∘ g) (g ⁻¹' s) b :=
  hf.comp_tendsto (tendsto_principal_principal.mpr <| Subset.refl _)


theorem IsExtrOn.on_preimage (g : δ → α) {b : δ} (hf : IsExtrOn f s (g b)) :
    IsExtrOn (f ∘ g) (g ⁻¹' s) b :=
  hf.elim (fun hf => (hf.on_preimage g).isExtr) fun hf => (hf.on_preimage g).isExtr


theorem IsMinOn.comp_mapsTo {t : Set δ} {g : δ → α} {b : δ} (hf : IsMinOn f s a) (hg : MapsTo g t s)
    (ha : g b = a) : IsMinOn (f ∘ g) t b := fun y hy => by
  /-
    α : Type u
    β : Type v
    δ : Type x
    inst✝ : Preorder β
    f : α → β
    s : Set α
    a : α
    t : Set δ
    g : δ → α
    b : δ
    hf : IsMinOn f s a
    hg : Set.MapsTo g t s
    ha : Eq (g b) a
    y : δ
    hy : Membership.mem t y
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Function.comp f g b) (Functi …
  -/
  simpa only [ha, (· ∘ ·)] using hf (hg hy)
  /-
    🎉 no goals
  -/


theorem IsMaxOn.comp_mapsTo {t : Set δ} {g : δ → α} {b : δ} (hf : IsMaxOn f s a) (hg : MapsTo g t s)
    (ha : g b = a) : IsMaxOn (f ∘ g) t b :=
  hf.dual.comp_mapsTo hg ha


theorem IsExtrOn.comp_mapsTo {t : Set δ} {g : δ → α} {b : δ} (hf : IsExtrOn f s a)
    (hg : MapsTo g t s) (ha : g b = a) : IsExtrOn (f ∘ g) t b :=
  hf.elim (fun h => Or.inl <| h.comp_mapsTo hg ha) fun h => Or.inr <| h.comp_mapsTo hg ha


theorem IsMinFilter.add (hf : IsMinFilter f l a) (hg : IsMinFilter g l a) :
    IsMinFilter (fun x => f x + g x) l a :=
  show IsMinFilter (fun x => f x + g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => add_le_add hx hy) hg


theorem IsMaxFilter.add (hf : IsMaxFilter f l a) (hg : IsMaxFilter g l a) :
    IsMaxFilter (fun x => f x + g x) l a :=
  show IsMaxFilter (fun x => f x + g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => add_le_add hx hy) hg


theorem IsMinOn.add (hf : IsMinOn f s a) (hg : IsMinOn g s a) : IsMinOn (fun x => f x + g x) s a :=
  IsMinFilter.add hf hg


theorem IsMaxOn.add (hf : IsMaxOn f s a) (hg : IsMaxOn g s a) : IsMaxOn (fun x => f x + g x) s a :=
  IsMaxFilter.add hf hg


theorem IsMinFilter.neg (hf : IsMinFilter f l a) : IsMaxFilter (fun x => -f x) l a :=
  hf.comp_antitone fun _x _y hx => neg_le_neg hx


theorem IsMaxFilter.neg (hf : IsMaxFilter f l a) : IsMinFilter (fun x => -f x) l a :=
  hf.comp_antitone fun _x _y hx => neg_le_neg hx


theorem IsExtrFilter.neg (hf : IsExtrFilter f l a) : IsExtrFilter (fun x => -f x) l a :=
  hf.elim (fun hf => hf.neg.isExtr) fun hf => hf.neg.isExtr


theorem IsMinOn.neg (hf : IsMinOn f s a) : IsMaxOn (fun x => -f x) s a :=
  hf.comp_antitone fun _x _y hx => neg_le_neg hx


theorem IsMaxOn.neg (hf : IsMaxOn f s a) : IsMinOn (fun x => -f x) s a :=
  hf.comp_antitone fun _x _y hx => neg_le_neg hx


theorem IsExtrOn.neg (hf : IsExtrOn f s a) : IsExtrOn (fun x => -f x) s a :=
  hf.elim (fun hf => hf.neg.isExtr) fun hf => hf.neg.isExtr


theorem IsMinFilter.sub (hf : IsMinFilter f l a) (hg : IsMaxFilter g l a) :
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 inst✝ : OrderedAddCommGroup β
                                                 f g : α → β
                                                 a : α
                                                 l : Filter α
                                                 hf : IsMinFilter f l a
                                                 hg : IsMaxFilter g l a
                                                 ⊢ IsMinFilter (fun x => HSub.hSub (f x) (g x)) l a
                                               -/
    IsMinFilter (fun x => f x - g x) l a := by simpa only [sub_eq_add_neg] using hf.add hg.neg
                                               /-
                                                 🎉 no goals
                                               -/


theorem IsMaxFilter.sub (hf : IsMaxFilter f l a) (hg : IsMinFilter g l a) :
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 inst✝ : OrderedAddCommGroup β
                                                 f g : α → β
                                                 a : α
                                                 l : Filter α
                                                 hf : IsMaxFilter f l a
                                                 hg : IsMinFilter g l a
                                                 ⊢ IsMaxFilter (fun x => HSub.hSub (f x) (g x)) l a
                                               -/
    IsMaxFilter (fun x => f x - g x) l a := by simpa only [sub_eq_add_neg] using hf.add hg.neg
                                               /-
                                                 🎉 no goals
                                               -/


theorem IsMinOn.sub (hf : IsMinOn f s a) (hg : IsMaxOn g s a) :
    IsMinOn (fun x => f x - g x) s a := by
  /-
    α : Type u
    β : Type v
    inst✝ : OrderedAddCommGroup β
    f g : α → β
    a : α
    s : Set α
    hf : IsMinOn f s a
    hg : IsMaxOn g s a
    ⊢ IsMinOn (fun x => HSub.hSub (f x) (g x)) s a
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem IsMaxOn.sub (hf : IsMaxOn f s a) (hg : IsMinOn g s a) :
    IsMaxOn (fun x => f x - g x) s a := by
  /-
    α : Type u
    β : Type v
    inst✝ : OrderedAddCommGroup β
    f g : α → β
    a : α
    s : Set α
    hf : IsMaxOn f s a
    hg : IsMinOn g s a
    ⊢ IsMaxOn (fun x => HSub.hSub (f x) (g x)) s a
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem IsMinFilter.sup (hf : IsMinFilter f l a) (hg : IsMinFilter g l a) :
    IsMinFilter (fun x => f x ⊔ g x) l a :=
  show IsMinFilter (fun x => f x ⊔ g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => sup_le_sup hx hy) hg


theorem IsMaxFilter.sup (hf : IsMaxFilter f l a) (hg : IsMaxFilter g l a) :
    IsMaxFilter (fun x => f x ⊔ g x) l a :=
  show IsMaxFilter (fun x => f x ⊔ g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => sup_le_sup hx hy) hg


theorem IsMinOn.sup (hf : IsMinOn f s a) (hg : IsMinOn g s a) : IsMinOn (fun x => f x ⊔ g x) s a :=
  IsMinFilter.sup hf hg


theorem IsMaxOn.sup (hf : IsMaxOn f s a) (hg : IsMaxOn g s a) : IsMaxOn (fun x => f x ⊔ g x) s a :=
  IsMaxFilter.sup hf hg


theorem IsMinFilter.inf (hf : IsMinFilter f l a) (hg : IsMinFilter g l a) :
    IsMinFilter (fun x => f x ⊓ g x) l a :=
  show IsMinFilter (fun x => f x ⊓ g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => inf_le_inf hx hy) hg


theorem IsMaxFilter.inf (hf : IsMaxFilter f l a) (hg : IsMaxFilter g l a) :
    IsMaxFilter (fun x => f x ⊓ g x) l a :=
  show IsMaxFilter (fun x => f x ⊓ g x) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => inf_le_inf hx hy) hg


theorem IsMinOn.inf (hf : IsMinOn f s a) (hg : IsMinOn g s a) : IsMinOn (fun x => f x ⊓ g x) s a :=
  IsMinFilter.inf hf hg


theorem IsMaxOn.inf (hf : IsMaxOn f s a) (hg : IsMaxOn g s a) : IsMaxOn (fun x => f x ⊓ g x) s a :=
  IsMaxFilter.inf hf hg


theorem IsMinFilter.min (hf : IsMinFilter f l a) (hg : IsMinFilter g l a) :
    IsMinFilter (fun x => min (f x) (g x)) l a :=
  show IsMinFilter (fun x => Min.min (f x) (g x)) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => min_le_min hx hy) hg


theorem IsMaxFilter.min (hf : IsMaxFilter f l a) (hg : IsMaxFilter g l a) :
    IsMaxFilter (fun x => min (f x) (g x)) l a :=
  show IsMaxFilter (fun x => Min.min (f x) (g x)) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => min_le_min hx hy) hg


theorem IsMinOn.min (hf : IsMinOn f s a) (hg : IsMinOn g s a) :
    IsMinOn (fun x => min (f x) (g x)) s a :=
  IsMinFilter.min hf hg


theorem IsMaxOn.min (hf : IsMaxOn f s a) (hg : IsMaxOn g s a) :
    IsMaxOn (fun x => min (f x) (g x)) s a :=
  IsMaxFilter.min hf hg


theorem IsMinFilter.max (hf : IsMinFilter f l a) (hg : IsMinFilter g l a) :
    IsMinFilter (fun x => max (f x) (g x)) l a :=
  show IsMinFilter (fun x => Max.max (f x) (g x)) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => max_le_max hx hy) hg


theorem IsMaxFilter.max (hf : IsMaxFilter f l a) (hg : IsMaxFilter g l a) :
    IsMaxFilter (fun x => max (f x) (g x)) l a :=
  show IsMaxFilter (fun x => Max.max (f x) (g x)) l a from
    hf.bicomp_mono (fun _x _x' hx _y _y' hy => max_le_max hx hy) hg


theorem IsMinOn.max (hf : IsMinOn f s a) (hg : IsMinOn g s a) :
    IsMinOn (fun x => max (f x) (g x)) s a :=
  IsMinFilter.max hf hg


theorem IsMaxOn.max (hf : IsMaxOn f s a) (hg : IsMaxOn g s a) :
    IsMaxOn (fun x => max (f x) (g x)) s a :=
  IsMaxFilter.max hf hg


theorem Filter.EventuallyLE.isMaxFilter {α β : Type*} [Preorder β] {f g : α → β} {a : α}
    {l : Filter α} (hle : g ≤ᶠ[l] f) (hfga : f a = g a) (h : IsMaxFilter f l a) :
    IsMaxFilter g l a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f g : α → β
    a : α
    l : Filter α
    hle : l.EventuallyLE g f
    hfga : Eq (f a) (g a)
    h : IsMaxFilter f l a
    ⊢ IsMaxFilter g l a
  -/
  refine hle.mp (h.mono fun x hf hgf => ?_)
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f g : α → β
    a : α
    l : Filter α
    hle : l.EventuallyLE g f
    hfga : Eq (f a) (g a)
    h : IsMaxFilter f l a
    x : α
    hf : LE.le (f x) (f a)
    hgf : LE.le (g x) (f x)
    ⊢ LE.le (g x) (g a)
  -/
  rw [← hfga]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f g : α → β
    a : α
    l : Filter α
    hle : l.EventuallyLE g f
    hfga : Eq (f a) (g a)
    h : IsMaxFilter f l a
    x : α
    hf : LE.le (f x) (f a)
    hgf : LE.le (g x) (f x)
    ⊢ LE.le (g x) (f a)
  -/
  exact le_trans hgf hf
  /-
    🎉 no goals
  -/


theorem IsMaxFilter.congr {α β : Type*} [Preorder β] {f g : α → β} {a : α} {l : Filter α}
    (h : IsMaxFilter f l a) (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsMaxFilter g l a :=
  heq.symm.le.isMaxFilter hfga h


theorem Filter.EventuallyEq.isMaxFilter_iff {α β : Type*} [Preorder β] {f g : α → β} {a : α}
    {l : Filter α} (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsMaxFilter f l a ↔ IsMaxFilter g l a :=
  ⟨fun h => h.congr heq hfga, fun h => h.congr heq.symm hfga.symm⟩


theorem Filter.EventuallyLE.isMinFilter {α β : Type*} [Preorder β] {f g : α → β} {a : α}
    {l : Filter α} (hle : f ≤ᶠ[l] g) (hfga : f a = g a) (h : IsMinFilter f l a) :
    IsMinFilter g l a :=
  @Filter.EventuallyLE.isMaxFilter _ βᵒᵈ _ _ _ _ _ hle hfga h


theorem IsMinFilter.congr {α β : Type*} [Preorder β] {f g : α → β} {a : α} {l : Filter α}
    (h : IsMinFilter f l a) (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsMinFilter g l a :=
  heq.le.isMinFilter hfga h


theorem Filter.EventuallyEq.isMinFilter_iff {α β : Type*} [Preorder β] {f g : α → β} {a : α}
    {l : Filter α} (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsMinFilter f l a ↔ IsMinFilter g l a :=
  ⟨fun h => h.congr heq hfga, fun h => h.congr heq.symm hfga.symm⟩


theorem IsExtrFilter.congr {α β : Type*} [Preorder β] {f g : α → β} {a : α} {l : Filter α}
    (h : IsExtrFilter f l a) (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsExtrFilter g l a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f g : α → β
    a : α
    l : Filter α
    h : IsExtrFilter f l a
    heq : l.EventuallyEq f g
    hfga : Eq (f a) (g a)
    ⊢ IsExtrFilter g l a
  -/
  rw [IsExtrFilter] at *
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Preorder β
    f g : α → β
    a : α
    l : Filter α
    h : Or (IsMinFilter f l a) (IsMaxFilter f l a)
    heq : l.EventuallyEq f g
    hfga : Eq (f a) (g a)
    ⊢ Or (IsMinFilter g l a) (IsMaxFilter g l a)
  -/
  rwa [← heq.isMaxFilter_iff hfga, ← heq.isMinFilter_iff hfga]
  /-
    🎉 no goals
  -/


theorem Filter.EventuallyEq.isExtrFilter_iff {α β : Type*} [Preorder β] {f g : α → β} {a : α}
    {l : Filter α} (heq : f =ᶠ[l] g) (hfga : f a = g a) : IsExtrFilter f l a ↔ IsExtrFilter g l a :=
  ⟨fun h => h.congr heq hfga, fun h => h.congr heq.symm hfga.symm⟩


theorem IsMaxOn.iSup_eq (hx₀ : x₀ ∈ s) (h : IsMaxOn f s x₀) : ⨆ x : s, f x = f x₀ :=
  haveI : Nonempty s := ⟨⟨x₀, hx₀⟩⟩
  ciSup_eq_of_forall_le_of_forall_lt_exists_gt (fun x => h x.2) fun _w hw => ⟨⟨x₀, hx₀⟩, hw⟩


theorem IsMinOn.iInf_eq (hx₀ : x₀ ∈ s) (h : IsMinOn f s x₀) : ⨅ x : s, f x = f x₀ :=
  @IsMaxOn.iSup_eq αᵒᵈ β _ _ _ _ hx₀ h


theorem sup_eq_of_isMaxOn {a : α} (hmem : a ∈ s) (hmax : IsMaxOn D s a) : s.sup D = D a :=
  (Finset.sup_le hmax).antisymm (Finset.le_sup hmem)


theorem sup_eq_of_max [Nonempty α] {b : β} (hb : b ∈ Set.range D) (hmem : D.invFun b ∈ s)
    (hmax : ∀ a ∈ s, D a ≤ b) : s.sup D = b := by
  /-
    α : Type u
    β : Type v
    inst✝² : SemilatticeSup β
    inst✝¹ : OrderBot β
    D : α → β
    s : Finset α
    inst✝ : Nonempty α
    b : β
    hb : Membership.mem (Set.range D) b
    hmem : Membership.mem s (Function.invFun D b)
    hmax : ∀ (a : α), Membership.mem s a → LE.le (D a) b
    ⊢ Eq (s.sup D) b
  -/
  obtain ⟨a, rfl⟩ := hb
  /-
    case intro
    α : Type u
    β : Type v
    inst✝² : SemilatticeSup β
    inst✝¹ : OrderBot β
    D : α → β
    s : Finset α
    inst✝ : Nonempty α
    a : α
    hmem : Membership.mem s (Function.invFun D (D a))
    hmax : ∀ (a_1 : α), Membership.mem s a_1 → LE.le (D a_1) (D a)
    ⊢ Eq (s.sup D) (D a)
  -/
  rw [← Function.apply_invFun_apply (f := D)]
  /-
    case intro
    α : Type u
    β : Type v
    inst✝² : SemilatticeSup β
    inst✝¹ : OrderBot β
    D : α → β
    s : Finset α
    inst✝ : Nonempty α
    a : α
    hmem : Membership.mem s (Function.invFun D (D a))
    hmax : ∀ (a_1 : α), Membership.mem s a_1 → LE.le (D a_1) (D a)
    ⊢ Eq (s.sup D) (D (Function.invFun D (D a)))
  -/
  apply sup_eq_of_isMaxOn hmem; intro
  /-
    case intro
    α : Type u
    β : Type v
    inst✝² : SemilatticeSup β
    inst✝¹ : OrderBot β
    D : α → β
    s : Finset α
    inst✝ : Nonempty α
    a : α
    hmem : Membership.mem s (Function.invFun D (D a))
    hmax : ∀ (a_1 : α), Membership.mem s a_1 → LE.le (D a_1) (D a)
    a✝ : α
    ⊢ Membership.mem (↑s) a✝ → Membership.mem (setOf fun x => (fun x => LE.le (D x …
  -/
  rw [Function.apply_invFun_apply (f := D)]; apply hmax
                                             /-
                                               🎉 no goals
                                             -/


theorem inf_eq_of_isMinOn {a : α} (hmem : a ∈ s) (hmax : IsMinOn D s a) : s.inf D = D a :=
  sup_eq_of_isMaxOn (α := αᵒᵈ) (β := βᵒᵈ) hmem hmax.dual


theorem inf_eq_of_min [Nonempty α] {b : β} (hb : b ∈ Set.range D) (hmem : D.invFun b ∈ s)
    (hmin : ∀ a ∈ s, b ≤ D a) : s.inf D = b :=
  sup_eq_of_max (α := αᵒᵈ) (β := βᵒᵈ) hb hmem hmin


