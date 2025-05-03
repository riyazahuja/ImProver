/-- A `Filtration` on a measurable space `Ω` with σ-algebra `m` is a monotone
sequence of sub-σ-algebras of `m`. -/
structure Filtration {Ω : Type*} (ι : Type*) [Preorder ι] (m : MeasurableSpace Ω) where
  seq : ι → MeasurableSpace Ω
  mono' : Monotone seq
  le' : ∀ i : ι, seq i ≤ m


instance [Preorder ι] : CoeFun (Filtration ι m) fun _ => ι → MeasurableSpace Ω :=
  ⟨fun f => f.seq⟩


protected theorem mono {i j : ι} (f : Filtration ι m) (hij : i ≤ j) : f i ≤ f j :=
  f.mono' hij


protected theorem le (f : Filtration ι m) (i : ι) : f i ≤ m :=
  f.le' i


@[ext]
protected theorem ext {f g : Filtration ι m} (h : (f : ι → MeasurableSpace Ω) = g) : f = g := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝ : Preorder ι
    f g : MeasureTheory.Filtration ι m
    h : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  cases f; cases g; congr
                    /-
                      🎉 no goals
                    -/


/-- The constant filtration which is equal to `m` for all `i : ι`. -/
def const (m' : MeasurableSpace Ω) (hm' : m' ≤ m) : Filtration ι m :=
  ⟨fun _ => m', monotone_const, fun _ => hm'⟩


@[simp]
theorem const_apply {m' : MeasurableSpace Ω} {hm' : m' ≤ m} (i : ι) : const ι m' hm' i = m' :=
  rfl


instance : Inhabited (Filtration ι m) :=
  ⟨const ι m le_rfl⟩


instance : LE (Filtration ι m) :=
  ⟨fun f g => ∀ i, f i ≤ g i⟩


instance : Bot (Filtration ι m) :=
  ⟨const ι ⊥ bot_le⟩


instance : Top (Filtration ι m) :=
  ⟨const ι m le_rfl⟩


instance : Max (Filtration ι m) :=
  ⟨fun f g =>
    { seq := fun i => f i ⊔ g i
      mono' := fun _ _ hij =>
        sup_le ((f.mono hij).trans le_sup_left) ((g.mono hij).trans le_sup_right)
      le' := fun i => sup_le (f.le i) (g.le i) }⟩


@[norm_cast]
theorem coeFn_sup {f g : Filtration ι m} : ⇑(f ⊔ g) = ⇑f ⊔ ⇑g :=
  rfl


instance : Min (Filtration ι m) :=
  ⟨fun f g =>
    { seq := fun i => f i ⊓ g i
      mono' := fun _ _ hij =>
        le_inf (inf_le_left.trans (f.mono hij)) (inf_le_right.trans (g.mono hij))
      le' := fun i => inf_le_left.trans (f.le i) }⟩


@[norm_cast]
theorem coeFn_inf {f g : Filtration ι m} : ⇑(f ⊓ g) = ⇑f ⊓ ⇑g :=
  rfl


instance : SupSet (Filtration ι m) :=
  ⟨fun s =>
    { seq := fun i => sSup ((fun f : Filtration ι m => f i) '' s)
      mono' := fun i j hij => by
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          ⊢ LE.le ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) i) ((fun i => Su …
        -/
        refine sSup_le fun m' hm' => ?_
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          hm' : Membership.mem (Set.image (fun f => ↑f i) s) m'
          ⊢ LE.le m' ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        rw [Set.mem_image] at hm'
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          hm' : Exists fun x => And (Membership.mem s x) (Eq (↑x i) m')
          ⊢ LE.le m' ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        obtain ⟨f, hf_mem, hfm'⟩ := hm'
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          hfm' : Eq (↑f i) m'
          ⊢ LE.le m' ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        rw [← hfm']
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          hfm' : Eq (↑f i) m'
          ⊢ LE.le (↑f i) ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        refine (f.mono hij).trans ?_
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          hfm' : Eq (↑f i) m'
          ⊢ LE.le (↑f j) ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        have hfj_mem : f j ∈ (fun g : Filtration ι m => g j) '' s := ⟨f, hf_mem, rfl⟩
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          hfm' : Eq (↑f i) m'
          hfj_mem : Membership.mem (Set.image (fun g => ↑g j) s) (↑f j)
          ⊢ LE.le (↑f j) ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) j)
        -/
        exact le_sSup hfj_mem
        /-
          🎉 no goals
        -/
      le' := fun i => by
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          ⊢ LE.le ((fun i => SupSet.sSup (Set.image (fun f => ↑f i) s)) i) m
        -/
        refine sSup_le fun m' hm' => ?_
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          m' : MeasurableSpace Ω
          hm' : Membership.mem (Set.image (fun f => ↑f i) s) m'
          ⊢ LE.le m' m
        -/
        rw [Set.mem_image] at hm'
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          m' : MeasurableSpace Ω
          hm' : Exists fun x => And (Membership.mem s x) (Eq (↑x i) m')
          ⊢ LE.le m' m
        -/
        obtain ⟨f, _, hfm'⟩ := hm'
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          left✝ : Membership.mem s f
          hfm' : Eq (↑f i) m'
          ⊢ LE.le m' m
        -/
        rw [← hfm']
        /-
          case intro.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          m' : MeasurableSpace Ω
          f : MeasureTheory.Filtration ι m
          left✝ : Membership.mem s f
          hfm' : Eq (↑f i) m'
          ⊢ LE.le (↑f i) m
        -/
        exact f.le i }⟩
        /-
          🎉 no goals
        -/


theorem sSup_def (s : Set (Filtration ι m)) (i : ι) :
    sSup s i = sSup ((fun f : Filtration ι m => f i) '' s) :=
  rfl


noncomputable instance : InfSet (Filtration ι m) :=
  ⟨fun s =>
    { seq := fun i => if Set.Nonempty s then sInf ((fun f : Filtration ι m => f i) '' s) else m
      mono' := fun i j hij => by
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          ⊢ LE.le ((fun i => ite s.Nonempty (InfSet.sInf (Set.image (fun f => ↑f i) s))  …
        -/
        by_cases h_nonempty : Set.Nonempty s
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          h_nonempty : s.Nonempty
          ⊢ LE.le ((fun i => ite s.Nonempty (InfSet.sInf (Set.image (fun f => ↑f i) s))  …
        -/
        swap; · simp only [h_nonempty, Set.image_nonempty, if_false, le_refl]
                /-
                  🎉 no goals
                -/
        simp only [h_nonempty, if_true, le_sInf_iff, Set.mem_image, forall_exists_index, and_imp,
          forall_apply_eq_imp_iff₂]
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          h_nonempty : s.Nonempty
          ⊢ ∀ (a : MeasureTheory.Filtration ι m), Membership.mem s a → LE.le (InfSet.sIn …
        -/
        refine fun f hf_mem => le_trans ?_ (f.mono hij)
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          h_nonempty : s.Nonempty
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          ⊢ LE.le (InfSet.sInf (Set.image (fun f => ↑f i) s)) (↑f i)
        -/
        have hfi_mem : f i ∈ (fun g : Filtration ι m => g i) '' s := ⟨f, hf_mem, rfl⟩
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i j : ι
          hij : LE.le i j
          h_nonempty : s.Nonempty
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          hfi_mem : Membership.mem (Set.image (fun g => ↑g i) s) (↑f i)
          ⊢ LE.le (InfSet.sInf (Set.image (fun f => ↑f i) s)) (↑f i)
        -/
        exact sInf_le hfi_mem
        /-
          🎉 no goals
        -/
      le' := fun i => by
        /-
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          ⊢ LE.le ((fun i => ite s.Nonempty (InfSet.sInf (Set.image (fun f => ↑f i) s))  …
        -/
        by_cases h_nonempty : Set.Nonempty s
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          h_nonempty : s.Nonempty
          ⊢ LE.le ((fun i => ite s.Nonempty (InfSet.sInf (Set.image (fun f => ↑f i) s))  …
        -/
        swap; · simp only [h_nonempty, if_false, le_refl]
                /-
                  🎉 no goals
                -/
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          h_nonempty : s.Nonempty
          ⊢ LE.le ((fun i => ite s.Nonempty (InfSet.sInf (Set.image (fun f => ↑f i) s))  …
        -/
        simp only [h_nonempty, if_true]
        /-
          case pos
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          h_nonempty : s.Nonempty
          ⊢ LE.le (InfSet.sInf (Set.image (fun f => ↑f i) s)) m
        -/
        obtain ⟨f, hf_mem⟩ := h_nonempty
        /-
          case pos.intro
          Ω : Type u_1
          β : Type u_2
          ι : Type u_3
          m : MeasurableSpace Ω
          inst✝ : Preorder ι
          s : Set (MeasureTheory.Filtration ι m)
          i : ι
          f : MeasureTheory.Filtration ι m
          hf_mem : Membership.mem s f
          ⊢ LE.le (InfSet.sInf (Set.image (fun f => ↑f i) s)) m
        -/
        exact le_trans (sInf_le ⟨f, hf_mem, rfl⟩) (f.le i) }⟩
        /-
          🎉 no goals
        -/


theorem sInf_def (s : Set (Filtration ι m)) (i : ι) :
    sInf s i = if Set.Nonempty s then sInf ((fun f : Filtration ι m => f i) '' s) else m :=
  rfl


noncomputable instance instCompleteLattice : CompleteLattice (Filtration ι m) where
  le := (· ≤ ·)
  le_refl _ _ := le_rfl
  le_trans _ _ _ h_fg h_gh i := (h_fg i).trans (h_gh i)
  le_antisymm _ _ h_fg h_gf := Filtration.ext <| funext fun i => (h_fg i).antisymm (h_gf i)
  sup := (· ⊔ ·)
  le_sup_left _ _ _ := le_sup_left
  le_sup_right _ _ _ := le_sup_right
  sup_le _ _ _ h_fh h_gh i := sup_le (h_fh i) (h_gh _)
  inf := (· ⊓ ·)
  inf_le_left _ _ _ := inf_le_left
  inf_le_right _ _ _ := inf_le_right
  le_inf _ _ _ h_fg h_fh i := le_inf (h_fg i) (h_fh i)
  sSup := sSup
  le_sSup _ f hf_mem _ := le_sSup ⟨f, hf_mem, rfl⟩
  sSup_le s f h_forall i :=
    sSup_le fun m' hm' => by
      /-
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        s : Set (MeasureTheory.Filtration ι m)
        f : MeasureTheory.Filtration ι m
        h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le b f
        i : ι
        m' : MeasurableSpace Ω
        hm' : Membership.mem (Set.image (fun f => ↑f i) s) m'
        ⊢ LE.le m' (↑f i)
      -/
      obtain ⟨g, hg_mem, hfm'⟩ := hm'
      /-
        case intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        s : Set (MeasureTheory.Filtration ι m)
        f : MeasureTheory.Filtration ι m
        h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le b f
        i : ι
        m' : MeasurableSpace Ω
        g : MeasureTheory.Filtration ι m
        hg_mem : Membership.mem s g
        hfm' : Eq ((fun f => ↑f i) g) m'
        ⊢ LE.le m' (↑f i)
      -/
      rw [← hfm']
      /-
        case intro.intro
        Ω : Type u_1
        β : Type u_2
        ι : Type u_3
        m : MeasurableSpace Ω
        inst✝ : Preorder ι
        s : Set (MeasureTheory.Filtration ι m)
        f : MeasureTheory.Filtration ι m
        h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le b f
        i : ι
        m' : MeasurableSpace Ω
        g : MeasureTheory.Filtration ι m
        hg_mem : Membership.mem s g
        hfm' : Eq ((fun f => ↑f i) g) m'
        ⊢ LE.le ((fun f => ↑f i) g) (↑f i)
      -/
      exact h_forall g hg_mem i
      /-
        🎉 no goals
      -/
  sInf := sInf
  sInf_le s f hf_mem i := by
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      hf_mem : Membership.mem s f
      i : ι
      ⊢ LE.le (↑(InfSet.sInf s) i) (↑f i)
    -/
    have hs : s.Nonempty := ⟨f, hf_mem⟩
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      hf_mem : Membership.mem s f
      i : ι
      hs : s.Nonempty
      ⊢ LE.le (↑(InfSet.sInf s) i) (↑f i)
    -/
    simp only [sInf_def, hs, if_true]
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      hf_mem : Membership.mem s f
      i : ι
      hs : s.Nonempty
      ⊢ LE.le (InfSet.sInf (Set.image (fun f => ↑f i) s)) (↑f i)
    -/
    exact sInf_le ⟨f, hf_mem, rfl⟩
    /-
      🎉 no goals
    -/
  le_sInf s f h_forall i := by
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le f b
      i : ι
      ⊢ LE.le (↑f i) (↑(InfSet.sInf s) i)
    -/
    by_cases hs : s.Nonempty
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le f b
      i : ι
      hs : s.Nonempty
      ⊢ LE.le (↑f i) (↑(InfSet.sInf s) i)
    -/
    swap; · simp only [sInf_def, hs, if_false]; exact f.le i
                                                /-
                                                  🎉 no goals
                                                -/
    simp only [sInf_def, hs, if_true, le_sInf_iff, Set.mem_image, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂]
    /-
      case pos
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝ : Preorder ι
      s : Set (MeasureTheory.Filtration ι m)
      f : MeasureTheory.Filtration ι m
      h_forall : ∀ (b : MeasureTheory.Filtration ι m), Membership.mem s b → LE.le f b
      i : ι
      hs : s.Nonempty
      ⊢ ∀ (a : MeasureTheory.Filtration ι m), Membership.mem s a → LE.le (↑f i) (↑a i)
    -/
    exact fun g hg_mem => h_forall g hg_mem i
    /-
      🎉 no goals
    -/
  top := ⊤
  bot := ⊥
  le_top f i := f.le' i
  bot_le _ _ := bot_le


theorem measurableSet_of_filtration [Preorder ι] {f : Filtration ι m} {s : Set Ω} {i : ι}
    (hs : MeasurableSet[f i] s) : MeasurableSet[m] s :=
  f.le i s hs


/-- A measure is σ-finite with respect to filtration if it is σ-finite with respect
to all the sub-σ-algebra of the filtration. -/
class SigmaFiniteFiltration [Preorder ι] (μ : Measure Ω) (f : Filtration ι m) : Prop where
  SigmaFinite : ∀ i : ι, SigmaFinite (μ.trim (f.le i))


instance sigmaFinite_of_sigmaFiniteFiltration [Preorder ι] (μ : Measure Ω) (f : Filtration ι m)
    [hf : SigmaFiniteFiltration μ f] (i : ι) : SigmaFinite (μ.trim (f.le i)) :=
  hf.SigmaFinite _


instance (priority := 100) IsFiniteMeasure.sigmaFiniteFiltration [Preorder ι] (μ : Measure Ω)
    (f : Filtration ι m) [IsFiniteMeasure μ] : SigmaFiniteFiltration μ f :=
               /-
                 Ω : Type u_1
                 β : Type u_2
                 ι : Type u_3
                 m : MeasurableSpace Ω
                 inst✝¹ : Preorder ι
                 μ : MeasureTheory.Measure Ω
                 f : MeasureTheory.Filtration ι m
                 inst✝ : MeasureTheory.IsFiniteMeasure μ
                 n : ι
                 ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
               -/
  ⟨fun n => by infer_instance⟩
               /-
                 🎉 no goals
               -/


/-- Given an integrable function `g`, the conditional expectations of `g` with respect to a
filtration is uniformly integrable. -/
theorem Integrable.uniformIntegrable_condexp_filtration [Preorder ι] {μ : Measure Ω}
    [IsFiniteMeasure μ] {f : Filtration ι m} {g : Ω → ℝ} (hg : Integrable g μ) :
    UniformIntegrable (fun i => μ[g|f i]) 1 μ :=
  hg.uniformIntegrable_condexp f.le


/-- Given a sequence of measurable sets `(sₙ)`, `filtrationOfSet` is the smallest filtration
such that `sₙ` is measurable with respect to the `n`-th sub-σ-algebra in `filtrationOfSet`. -/
def filtrationOfSet {s : ι → Set Ω} (hsm : ∀ i, MeasurableSet (s i)) : Filtration ι m where
  seq i := MeasurableSpace.generateFrom {t | ∃ j ≤ i, s j = t}
  mono' _ _ hnm := MeasurableSpace.generateFrom_mono fun _ ⟨k, hk₁, hk₂⟩ => ⟨k, hk₁.trans hnm, hk₂⟩
  le' _ := MeasurableSpace.generateFrom_le fun _ ⟨k, _, hk₂⟩ => hk₂ ▸ hsm k


theorem measurableSet_filtrationOfSet {s : ι → Set Ω} (hsm : ∀ i, MeasurableSet[m] (s i)) (i : ι)
    {j : ι} (hj : j ≤ i) : MeasurableSet[filtrationOfSet hsm i] (s j) :=
  MeasurableSpace.measurableSet_generateFrom ⟨j, hj, rfl⟩


theorem measurableSet_filtrationOfSet' {s : ι → Set Ω} (hsm : ∀ n, MeasurableSet[m] (s n))
    (i : ι) : MeasurableSet[filtrationOfSet hsm i] (s i) :=
  measurableSet_filtrationOfSet hsm i le_rfl


/-- Given a sequence of functions, the natural filtration is the smallest sequence
of σ-algebras such that that sequence of functions is measurable with respect to
the filtration. -/
def natural (u : ι → Ω → β) (hum : ∀ i, StronglyMeasurable (u i)) : Filtration ι m where
  seq i := ⨆ j ≤ i, MeasurableSpace.comap (u j) mβ
  mono' _ _ hij := biSup_mono fun _ => ge_trans hij
  le' i := by
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝¹ : BorelSpace β
      inst✝ : Preorder ι
      u : ι → Ω → β
      hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
      i : ι
      ⊢ LE.le ((fun i => iSup fun j => iSup fun h => MeasurableSpace.comap (u j) mβ) …
    -/
    refine iSup₂_le ?_
    /-
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝¹ : BorelSpace β
      inst✝ : Preorder ι
      u : ι → Ω → β
      hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
      i : ι
      ⊢ ∀ (i_1 : ι), LE.le i_1 i → LE.le (MeasurableSpace.comap (u i_1) mβ) m
    -/
    rintro j _ s ⟨t, ht, rfl⟩
    /-
      case intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝¹ : BorelSpace β
      inst✝ : Preorder ι
      u : ι → Ω → β
      hum : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
      i j : ι
      j✝ : LE.le j i
      t : Set β
      ht : MeasurableSet t
      ⊢ MeasurableSet (Set.preimage (u j) t)
    -/
    exact (hum j).measurable ht
    /-
      🎉 no goals
    -/


theorem filtrationOfSet_eq_natural [MulZeroOneClass β] [Nontrivial β] {s : ι → Set Ω}
    (hsm : ∀ i, MeasurableSet[m] (s i)) :
    filtrationOfSet hsm = natural (fun i => (s i).indicator (fun _ => 1 : Ω → β)) fun i =>
      stronglyMeasurable_one.indicator (hsm i) := by
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝³ : BorelSpace β
    inst✝² : Preorder ι
    inst✝¹ : MulZeroOneClass β
    inst✝ : Nontrivial β
    s : ι → Set Ω
    hsm : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq (MeasureTheory.filtrationOfSet hsm) (MeasureTheory.Filtration.natural (fu …
  -/
  simp only [filtrationOfSet, natural, measurableSpace_iSup_eq, exists_prop, mk.injEq]
  /-
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝³ : BorelSpace β
    inst✝² : Preorder ι
    inst✝¹ : MulZeroOneClass β
    inst✝ : Nontrivial β
    s : ι → Set Ω
    hsm : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq (fun i => MeasurableSpace.generateFrom (setOf fun t => Exists fun j => An …
  -/
  ext1 i
  /-
    case h
    Ω : Type u_1
    β : Type u_2
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.MetrizableSpace β
    mβ : MeasurableSpace β
    inst✝³ : BorelSpace β
    inst✝² : Preorder ι
    inst✝¹ : MulZeroOneClass β
    inst✝ : Nontrivial β
    s : ι → Set Ω
    hsm : ∀ (i : ι), MeasurableSet (s i)
    i : ι
    ⊢ Eq (MeasurableSpace.generateFrom (setOf fun t => Exists fun j => And (LE.le  …
  -/
  refine le_antisymm (generateFrom_le ?_) (generateFrom_le ?_)
    /-
      case h.refine_1
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      ⊢ ∀ (t : Set Ω), Membership.mem (setOf fun t => Exists fun j => And (LE.le j i …
    -/
  · rintro _ ⟨j, hij, rfl⟩
    /-
      case h.refine_1.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i j : ι
      hij : LE.le j i
      ⊢ MeasurableSet (s j)
    -/
    refine measurableSet_generateFrom ⟨j, measurableSet_generateFrom ⟨hij, ?_⟩⟩
    /-
      case h.refine_1.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i j : ι
      hij : LE.le j i
      ⊢ MeasurableSet (s j)
    -/
    rw [comap_eq_generateFrom]
    /-
      case h.refine_1.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i j : ι
      hij : LE.le j i
      ⊢ MeasurableSet (s j)
    -/
    refine measurableSet_generateFrom ⟨{1}, measurableSet_singleton 1, ?_⟩
    /-
      case h.refine_1.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i j : ι
      hij : LE.le j i
      ⊢ Eq (Set.preimage ((s j).indicator fun x => 1) (Singleton.singleton 1)) (s j)
    -/
    ext x
    /-
      case h.refine_1.intro.intro.h
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i j : ι
      hij : LE.le j i
      x : Ω
      ⊢ Iff (Membership.mem (Set.preimage ((s j).indicator fun x => 1) (Singleton.si …
    -/
    simp [Set.indicator_const_preimage_eq_union]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      ⊢ ∀ (t : Set Ω), Membership.mem (setOf fun s_1 => Exists fun n => MeasurableSe …
    -/
  · rintro t ⟨n, ht⟩
    suffices MeasurableSpace.generateFrom {t | n ≤ i ∧
      MeasurableSet[MeasurableSpace.comap ((s n).indicator (fun _ => 1 : Ω → β)) mβ] t} ≤
        MeasurableSpace.generateFrom {t | ∃ (j : ι), j ≤ i ∧ s j = t} by
      exact this _ ht
    /-
      case h.refine_2.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      t : Set Ω
      n : ι
      ht : MeasurableSet t
      ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun t => And (LE.le n i) (Measura …
    -/
    refine generateFrom_le ?_
    /-
      case h.refine_2.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      t : Set Ω
      n : ι
      ht : MeasurableSet t
      ⊢ ∀ (t : Set Ω), Membership.mem (setOf fun t => And (LE.le n i) (MeasurableSet …
    -/
    rintro t ⟨hn, u, _, hu'⟩
    /-
      case h.refine_2.intro.intro.intro.intro
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      t✝ : Set Ω
      n : ι
      ht : MeasurableSet t✝
      t : Set Ω
      hn : LE.le n i
      u : Set β
      left✝ : MeasurableSet u
      hu' : Eq (Set.preimage ((s n).indicator fun x => 1) u) t
      ⊢ MeasurableSet t
    -/
    obtain heq | heq | heq | heq := Set.indicator_const_preimage (s n) u (1 : β)
    /-
      case h.refine_2.intro.intro.intro.intro.inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      t✝ : Set Ω
      n : ι
      ht : MeasurableSet t✝
      t : Set Ω
      hn : LE.le n i
      u : Set β
      left✝ : MeasurableSet u
      hu' : Eq (Set.preimage ((s n).indicator fun x => 1) u) t
      heq : Eq (Set.preimage ((s n).indicator fun x => 1) u) Set.univ
      ⊢ MeasurableSet t
    -/
    on_goal 4 => rw [Set.mem_singleton_iff] at heq
    /-
      case h.refine_2.intro.intro.intro.intro.inl
      Ω : Type u_1
      β : Type u_2
      ι : Type u_3
      m : MeasurableSpace Ω
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : TopologicalSpace.MetrizableSpace β
      mβ : MeasurableSpace β
      inst✝³ : BorelSpace β
      inst✝² : Preorder ι
      inst✝¹ : MulZeroOneClass β
      inst✝ : Nontrivial β
      s : ι → Set Ω
      hsm : ∀ (i : ι), MeasurableSet (s i)
      i : ι
      t✝ : Set Ω
      n : ι
      ht : MeasurableSet t✝
      t : Set Ω
      hn : LE.le n i
      u : Set β
      left✝ : MeasurableSet u
      hu' : Eq (Set.preimage ((s n).indicator fun x => 1) u) t
      heq : Eq (Set.preimage ((s n).indicator fun x => 1) u) Set.univ
      ⊢ MeasurableSet t
    -/
    all_goals rw [heq] at hu'; rw [← hu']
    exacts [MeasurableSet.univ, measurableSet_generateFrom ⟨n, hn, rfl⟩,
      MeasurableSet.compl (measurableSet_generateFrom ⟨n, hn, rfl⟩), measurableSet_empty _]


/-- Given a process `f` and a filtration `ℱ`, if `f` converges to some `g` almost everywhere and
`g` is `⨆ n, ℱ n`-measurable, then `limitProcess f ℱ μ` chooses said `g`, else it returns 0.

This definition is used to phrase the a.e. martingale convergence theorem
`Submartingale.ae_tendsto_limitProcess` where an L¹-bounded submartingale `f` adapted to `ℱ`
converges to `limitProcess f ℱ μ` `μ`-almost everywhere. -/
noncomputable def limitProcess (f : ι → Ω → E) (ℱ : Filtration ι m)
    (μ : Measure Ω) :=
  if h : ∃ g : Ω → E,
    StronglyMeasurable[⨆ n, ℱ n] g ∧ ∀ᵐ ω ∂μ, Tendsto (fun n => f n ω) atTop (𝓝 (g ω)) then
  Classical.choose h else 0


theorem stronglyMeasurable_limitProcess : StronglyMeasurable[⨆ n, ℱ n] (limitProcess f ℱ μ) := by
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : Preorder ι
    E : Type u_4
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    ℱ : MeasureTheory.Filtration ι m
    f : ι → Ω → E
    μ : MeasureTheory.Measure Ω
    ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.Filtration.limitProcess f ℱ μ)
  -/
  rw [limitProcess]
  /-
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : Preorder ι
    E : Type u_4
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    ℱ : MeasureTheory.Filtration ι m
    f : ι → Ω → E
    μ : MeasureTheory.Measure Ω
    ⊢ MeasureTheory.StronglyMeasurable (dite (Exists fun g => And (MeasureTheory.S …
  -/
  split_ifs with h
  /-
    case pos
    Ω : Type u_1
    ι : Type u_3
    m : MeasurableSpace Ω
    inst✝² : Preorder ι
    E : Type u_4
    inst✝¹ : Zero E
    inst✝ : TopologicalSpace E
    ℱ : MeasureTheory.Filtration ι m
    f : ι → Ω → E
    μ : MeasureTheory.Measure Ω
    h : Exists fun g => And (MeasureTheory.StronglyMeasurable g) (Filter.Eventuall …
    ⊢ MeasureTheory.StronglyMeasurable (Classical.choose h)
  -/
  exacts [(Classical.choose_spec h).1, stronglyMeasurable_zero]
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_limit_process' : StronglyMeasurable[m] (limitProcess f ℱ μ) :=
  stronglyMeasurable_limitProcess.mono (sSup_le fun _ ⟨_, hn⟩ => hn ▸ ℱ.le _)


theorem memℒp_limitProcess_of_eLpNorm_bdd {R : ℝ≥0} {p : ℝ≥0∞} {F : Type*} [NormedAddCommGroup F]
    {ℱ : Filtration ℕ m} {f : ℕ → Ω → F} (hfm : ∀ n, AEStronglyMeasurable (f n) μ)
    (hbdd : ∀ n, eLpNorm (f n) p μ ≤ R) : Memℒp (limitProcess f ℱ μ) p μ := by
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    p : ENNReal
    F : Type u_5
    inst✝ : NormedAddCommGroup F
    ℱ : MeasureTheory.Filtration Nat m
    f : Nat → Ω → F
    hfm : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    ⊢ MeasureTheory.Memℒp (MeasureTheory.Filtration.limitProcess f ℱ μ) p μ
  -/
  rw [limitProcess]
  /-
    Ω : Type u_1
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    R : NNReal
    p : ENNReal
    F : Type u_5
    inst✝ : NormedAddCommGroup F
    ℱ : MeasureTheory.Filtration Nat m
    f : Nat → Ω → F
    hfm : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
    ⊢ MeasureTheory.Memℒp (dite (Exists fun g => And (MeasureTheory.StronglyMeasur …
  -/
  split_ifs with h
  · refine ⟨StronglyMeasurable.aestronglyMeasurable
      ((Classical.choose_spec h).1.mono (sSup_le fun m ⟨n, hn⟩ => hn ▸ ℱ.le _)),
      lt_of_le_of_lt (Lp.eLpNorm_lim_le_liminf_eLpNorm hfm _ (Classical.choose_spec h).2)
        (lt_of_le_of_lt ?_ (ENNReal.coe_lt_top : ↑R < ∞))⟩
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      R : NNReal
      p : ENNReal
      F : Type u_5
      inst✝ : NormedAddCommGroup F
      ℱ : MeasureTheory.Filtration Nat m
      f : Nat → Ω → F
      hfm : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      h : Exists fun g => And (MeasureTheory.StronglyMeasurable g) (Filter.Eventuall …
      ⊢ LE.le (Filter.liminf (fun n => MeasureTheory.eLpNorm (f n) p μ) Filter.atTop …
    -/
    simp_rw [liminf_eq, eventually_atTop]
    /-
      case pos
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      R : NNReal
      p : ENNReal
      F : Type u_5
      inst✝ : NormedAddCommGroup F
      ℱ : MeasureTheory.Filtration Nat m
      f : Nat → Ω → F
      hfm : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      h : Exists fun g => And (MeasureTheory.StronglyMeasurable g) (Filter.Eventuall …
      ⊢ LE.le (SupSet.sSup (setOf fun a => Exists fun a_1 => ∀ (b : Nat), GE.ge b a_ …
    -/
    exact sSup_le fun b ⟨a, ha⟩ => (ha a le_rfl).trans (hbdd _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      R : NNReal
      p : ENNReal
      F : Type u_5
      inst✝ : NormedAddCommGroup F
      ℱ : MeasureTheory.Filtration Nat m
      f : Nat → Ω → F
      hfm : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) p μ) ↑R
      h : Not (Exists fun g => And (MeasureTheory.StronglyMeasurable g) (Filter.Even …
      ⊢ MeasureTheory.Memℒp 0 p μ
    -/
  · exact zero_memℒp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-07-27")]
alias memℒp_limitProcess_of_snorm_bdd := memℒp_limitProcess_of_eLpNorm_bdd


/-- The exterior σ-algebras of finite sets of `α` form a cofiltration indexed by `Finset α`. -/
def cylinderEventsCompl : Filtration (Finset α)ᵒᵈ (.pi (π := fun _ : α ↦ Ω)) where
  seq Λ := cylinderEvents (↑(OrderDual.ofDual Λ))ᶜ
  mono' _ _ h := cylinderEvents_mono <| Set.compl_subset_compl_of_subset h
  le' _  := cylinderEvents_le_pi


