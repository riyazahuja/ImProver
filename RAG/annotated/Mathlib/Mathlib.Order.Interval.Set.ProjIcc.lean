/-- Projection of `α` to the closed interval `[a, ∞)`. -/
def projIci (a x : α) : Ici a := ⟨max a x, le_max_left _ _⟩


/-- Projection of `α` to the closed interval `(-∞, b]`. -/
def projIic (b x : α) : Iic b := ⟨min b x, min_le_left _ _⟩


/-- Projection of `α` to the closed interval `[a, b]`. -/
def projIcc (a b : α) (h : a ≤ b) (x : α) : Icc a b :=
  ⟨max a (min b x), le_max_left _ _, max_le h (min_le_left _ _)⟩


@[norm_cast]
theorem coe_projIci (a x : α) : (projIci a x : α) = max a x := rfl


@[norm_cast]
theorem coe_projIic (b x : α) : (projIic b x : α) = min b x := rfl


@[norm_cast]
theorem coe_projIcc (a b : α) (h : a ≤ b) (x : α) : (projIcc a b h x : α) = max a (min b x) := rfl


theorem projIci_of_le (hx : x ≤ a) : projIci a x = ⟨a, le_rfl⟩ := Subtype.ext <| max_eq_left hx


theorem projIic_of_le (hx : b ≤ x) : projIic b x = ⟨b, le_rfl⟩ := Subtype.ext <| min_eq_left hx


theorem projIcc_of_le_left (hx : x ≤ a) : projIcc a b h x = ⟨a, left_mem_Icc.2 h⟩ := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    x : α
    hx : LE.le x a
    ⊢ Eq (Set.projIcc a b h x) ⟨a, ⋯⟩
  -/
  simp [projIcc, hx, hx.trans h]
  /-
    🎉 no goals
  -/



theorem projIcc_of_right_le (hx : b ≤ x) : projIcc a b h x = ⟨b, right_mem_Icc.2 h⟩ := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    x : α
    hx : LE.le b x
    ⊢ Eq (Set.projIcc a b h x) ⟨b, ⋯⟩
  -/
  simp [projIcc, hx, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem projIci_self (a : α) : projIci a a = ⟨a, le_rfl⟩ := projIci_of_le le_rfl


@[simp]
theorem projIic_self (b : α) : projIic b b = ⟨b, le_rfl⟩ := projIic_of_le le_rfl


@[simp]
theorem projIcc_left : projIcc a b h a = ⟨a, left_mem_Icc.2 h⟩ :=
  projIcc_of_le_left h le_rfl


@[simp]
theorem projIcc_right : projIcc a b h b = ⟨b, right_mem_Icc.2 h⟩ :=
  projIcc_of_right_le h le_rfl


                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : LinearOrder α
                                                                    a x : α
                                                                    ⊢ Iff (Eq (Set.projIci a x) ⟨a, ⋯⟩) (LE.le x a)
                                                                  -/
theorem projIci_eq_self : projIci a x = ⟨a, le_rfl⟩ ↔ x ≤ a := by simp [projIci, Subtype.ext_iff]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                                  /-
                                                                    α : Type u_1
                                                                    inst✝ : LinearOrder α
                                                                    b x : α
                                                                    ⊢ Iff (Eq (Set.projIic b x) ⟨b, ⋯⟩) (LE.le b x)
                                                                  -/
theorem projIic_eq_self : projIic b x = ⟨b, le_rfl⟩ ↔ b ≤ x := by simp [projIic, Subtype.ext_iff]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem projIcc_eq_left (h : a < b) : projIcc a b h.le x = ⟨a, left_mem_Icc.mpr h.le⟩ ↔ x ≤ a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b x : α
    h : LT.lt a b
    ⊢ Iff (Eq (Set.projIcc a b ⋯ x) ⟨a, ⋯⟩) (LE.le x a)
  -/
  simp [projIcc, Subtype.ext_iff, h.not_le]
  /-
    🎉 no goals
  -/


theorem projIcc_eq_right (h : a < b) : projIcc a b h.le x = ⟨b, right_mem_Icc.2 h.le⟩ ↔ b ≤ x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b x : α
    h : LT.lt a b
    ⊢ Iff (Eq (Set.projIcc a b ⋯ x) ⟨b, ⋯⟩) (LE.le b x)
  -/
  simp [projIcc, Subtype.ext_iff, max_min_distrib_left, h.le, h.not_le]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : LinearOrder α
                                                                        a x : α
                                                                        hx : Membership.mem (Set.Ici a) x
                                                                        ⊢ Eq (Set.projIci a x) ⟨x, hx⟩
                                                                      -/
theorem projIci_of_mem (hx : x ∈ Ici a) : projIci a x = ⟨x, hx⟩ := by simpa [projIci]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                                      /-
                                                                        α : Type u_1
                                                                        inst✝ : LinearOrder α
                                                                        b x : α
                                                                        hx : Membership.mem (Set.Iic b) x
                                                                        ⊢ Eq (Set.projIic b x) ⟨x, hx⟩
                                                                      -/
theorem projIic_of_mem (hx : x ∈ Iic b) : projIic b x = ⟨x, hx⟩ := by simpa [projIic]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem projIcc_of_mem (hx : x ∈ Icc a b) : projIcc a b h x = ⟨x, hx⟩ := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    x : α
    hx : Membership.mem (Set.Icc a b) x
    ⊢ Eq (Set.projIcc a b h x) ⟨x, hx⟩
  -/
  simp [projIcc, hx.1, hx.2]
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : LinearOrder α
                                                          a : α
                                                          x : ↑(Set.Ici a)
                                                          ⊢ Eq (Set.projIci a ↑x) x
                                                        -/
theorem projIci_coe (x : Ici a) : projIci a x = x := by cases x; apply projIci_of_mem
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : LinearOrder α
                                                          b : α
                                                          x : ↑(Set.Iic b)
                                                          ⊢ Eq (Set.projIic b ↑x) x
                                                        -/
theorem projIic_coe (x : Iic b) : projIic b x = x := by cases x; apply projIic_of_mem
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem projIcc_val (x : Icc a b) : projIcc a b h x = x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    x : ↑(Set.Icc a b)
    ⊢ Eq (Set.projIcc a b h ↑x) x
  -/
  cases x
  /-
    case mk
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    val✝ : α
    property✝ : Membership.mem (Set.Icc a b) val✝
    ⊢ Eq (Set.projIcc a b h ↑⟨val✝, property✝⟩) ⟨val✝, property✝⟩
  -/
  apply projIcc_of_mem
  /-
    🎉 no goals
  -/


theorem projIci_surjOn : SurjOn (projIci a) (Ici a) univ := fun x _ => ⟨x, x.2, projIci_coe x⟩


theorem projIic_surjOn : SurjOn (projIic b) (Iic b) univ := fun x _ => ⟨x, x.2, projIic_coe x⟩


theorem projIcc_surjOn : SurjOn (projIcc a b h) (Icc a b) univ := fun x _ =>
  ⟨x, x.2, projIcc_val h x⟩


theorem projIci_surjective : Surjective (projIci a) := fun x => ⟨x, projIci_coe x⟩


theorem projIic_surjective : Surjective (projIic b) := fun x => ⟨x, projIic_coe x⟩


theorem projIcc_surjective : Surjective (projIcc a b h) := fun x => ⟨x, projIcc_val h x⟩


@[simp]
theorem range_projIci : range (projIci a) = univ := projIci_surjective.range_eq


@[simp]
theorem range_projIic : range (projIic a) = univ := projIic_surjective.range_eq


@[simp]
theorem range_projIcc : range (projIcc a b h) = univ :=
  (projIcc_surjective h).range_eq


theorem monotone_projIci : Monotone (projIci a) := fun _ _ => max_le_max le_rfl


theorem monotone_projIic : Monotone (projIic a) := fun _ _ => min_le_min le_rfl


theorem monotone_projIcc : Monotone (projIcc a b h) := fun _ _ hxy =>
  max_le_max le_rfl <| min_le_min le_rfl hxy


theorem strictMonoOn_projIci : StrictMonoOn (projIci a) (Ici a) := fun x hx y hy hxy => by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a x : α
    hx : Membership.mem (Set.Ici a) x
    y : α
    hy : Membership.mem (Set.Ici a) y
    hxy : LT.lt x y
    ⊢ LT.lt (Set.projIci a x) (Set.projIci a y)
  -/
  simpa only [projIci_of_mem, hx, hy]
  /-
    🎉 no goals
  -/


theorem strictMonoOn_projIic : StrictMonoOn (projIic b) (Iic b) := fun x hx y hy hxy => by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    b x : α
    hx : Membership.mem (Set.Iic b) x
    y : α
    hy : Membership.mem (Set.Iic b) y
    hxy : LT.lt x y
    ⊢ LT.lt (Set.projIic b x) (Set.projIic b y)
  -/
  simpa only [projIic_of_mem, hx, hy]
  /-
    🎉 no goals
  -/


theorem strictMonoOn_projIcc : StrictMonoOn (projIcc a b h) (Icc a b) := fun x hx y hy hxy => by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    x : α
    hx : Membership.mem (Set.Icc a b) x
    y : α
    hy : Membership.mem (Set.Icc a b) y
    hxy : LT.lt x y
    ⊢ LT.lt (Set.projIcc a b h x) (Set.projIcc a b h y)
  -/
  simpa only [projIcc_of_mem, hx, hy]
  /-
    🎉 no goals
  -/


/-- Extend a function `[a, ∞) → β` to a map `α → β`. -/
def IciExtend (f : Ici a → β) : α → β :=
  f ∘ projIci a


/-- Extend a function `(-∞, b] → β` to a map `α → β`. -/
def IicExtend (f : Iic b → β) : α → β :=
  f ∘ projIic b


/-- Extend a function `[a, b] → β` to a map `α → β`. -/
def IccExtend {a b : α} (h : a ≤ b) (f : Icc a b → β) : α → β :=
  f ∘ projIcc a b h


theorem IciExtend_apply (f : Ici a → β) (x : α) : IciExtend f x = f ⟨max a x, le_max_left _ _⟩ :=
  rfl


theorem IicExtend_apply (f : Iic b → β) (x : α) : IicExtend f x = f ⟨min b x, min_le_left _ _⟩ :=
  rfl


theorem IccExtend_apply (h : a ≤ b) (f : Icc a b → β) (x : α) :
    IccExtend h f x = f ⟨max a (min b x), le_max_left _ _, max_le h (min_le_left _ _)⟩ := rfl


@[simp]
theorem range_IciExtend (f : Ici a → β) : range (IciExtend f) = range f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder α
    a : α
    f : ↑(Set.Ici a) → β
    ⊢ Eq (Set.range (Set.IciExtend f)) (Set.range f)
  -/
  simp only [IciExtend, range_comp f, range_projIci, range_id', image_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_IicExtend (f : Iic b → β) : range (IicExtend f) = range f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder α
    b : α
    f : ↑(Set.Iic b) → β
    ⊢ Eq (Set.range (Set.IicExtend f)) (Set.range f)
  -/
  simp only [IicExtend, range_comp f, range_projIic, range_id', image_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem IccExtend_range (f : Icc a b → β) : range (IccExtend h f) = range f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    f : ↑(Set.Icc a b) → β
    ⊢ Eq (Set.range (Set.IccExtend h f)) (Set.range f)
  -/
  simp only [IccExtend, range_comp f, range_projIcc, image_univ]
  /-
    🎉 no goals
  -/


theorem IciExtend_of_le (f : Ici a → β) (hx : x ≤ a) : IciExtend f x = f ⟨a, le_rfl⟩ :=
  congr_arg f <| projIci_of_le hx


theorem IicExtend_of_le (f : Iic b → β) (hx : b ≤ x) : IicExtend f x = f ⟨b, le_rfl⟩ :=
  congr_arg f <| projIic_of_le hx


theorem IccExtend_of_le_left (f : Icc a b → β) (hx : x ≤ a) :
    IccExtend h f x = f ⟨a, left_mem_Icc.2 h⟩ :=
  congr_arg f <| projIcc_of_le_left h hx


theorem IccExtend_of_right_le (f : Icc a b → β) (hx : b ≤ x) :
    IccExtend h f x = f ⟨b, right_mem_Icc.2 h⟩ :=
  congr_arg f <| projIcc_of_right_le h hx


@[simp]
theorem IciExtend_self (f : Ici a → β) : IciExtend f a = f ⟨a, le_rfl⟩ :=
  IciExtend_of_le f le_rfl


@[simp]
theorem IicExtend_self (f : Iic b → β) : IicExtend f b = f ⟨b, le_rfl⟩ :=
  IicExtend_of_le f le_rfl


@[simp]
theorem IccExtend_left (f : Icc a b → β) : IccExtend h f a = f ⟨a, left_mem_Icc.2 h⟩ :=
  IccExtend_of_le_left h f le_rfl


@[simp]
theorem IccExtend_right (f : Icc a b → β) : IccExtend h f b = f ⟨b, right_mem_Icc.2 h⟩ :=
  IccExtend_of_right_le h f le_rfl


theorem IciExtend_of_mem (f : Ici a → β) (hx : x ∈ Ici a) : IciExtend f x = f ⟨x, hx⟩ :=
  congr_arg f <| projIci_of_mem hx


theorem IicExtend_of_mem (f : Iic b → β) (hx : x ∈ Iic b) : IicExtend f x = f ⟨x, hx⟩ :=
  congr_arg f <| projIic_of_mem hx


theorem IccExtend_of_mem (f : Icc a b → β) (hx : x ∈ Icc a b) : IccExtend h f x = f ⟨x, hx⟩ :=
  congr_arg f <| projIcc_of_mem h hx


@[simp]
theorem IciExtend_coe (f : Ici a → β) (x : Ici a) : IciExtend f x = f x :=
  congr_arg f <| projIci_coe x


@[simp]
theorem IicExtend_coe (f : Iic b → β) (x : Iic b) : IicExtend f x = f x :=
  congr_arg f <| projIic_coe x


@[simp]
theorem IccExtend_val (f : Icc a b → β) (x : Icc a b) : IccExtend h f x = f x :=
  congr_arg f <| projIcc_val h x


/-- If `f : α → β` is a constant both on $(-∞, a]$ and on $[b, +∞)$, then the extension of this
function from $[a, b]$ to the whole line is equal to the original function. -/
theorem IccExtend_eq_self (f : α → β) (ha : ∀ x < a, f x = f a) (hb : ∀ x, b < x → f x = f b) :
    IccExtend h (f ∘ (↑)) = f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    f : α → β
    ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
    hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
    ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val)) f
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    f : α → β
    ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
    hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
    x : α
    ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) x) (f x)
  -/
  cases' lt_or_le x a with hxa hax
    /-
      case h.inl
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder α
      a b : α
      h : LE.le a b
      f : α → β
      ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
      hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
      x : α
      hxa : LT.lt x a
      ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) x) (f x)
    -/
  · simp [IccExtend_of_le_left _ _ hxa.le, ha x hxa]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      inst✝ : LinearOrder α
      a b : α
      h : LE.le a b
      f : α → β
      ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
      hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
      x : α
      hax : LE.le a x
      ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) x) (f x)
    -/
  · rcases le_or_lt x b with hxb | hbx
      /-
        case h.inr.inl
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder α
        a b : α
        h : LE.le a b
        f : α → β
        ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
        hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
        x : α
        hax : LE.le a x
        hxb : LE.le x b
        ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) x) (f x)
      -/
    · lift x to Icc a b using ⟨hax, hxb⟩
      /-
        case h.inr.inl.intro
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder α
        a b : α
        h : LE.le a b
        f : α → β
        ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
        hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
        x : Subtype fun x => Membership.mem (Set.Icc a b) x
        hax : LE.le a ↑x
        hxb : LE.le (↑x) b
        ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) ↑x) (f ↑x)
      -/
      rw [IccExtend_val, comp_apply]
      /-
        🎉 no goals
      -/
      /-
        case h.inr.inr
        α : Type u_1
        β : Type u_2
        inst✝ : LinearOrder α
        a b : α
        h : LE.le a b
        f : α → β
        ha : ∀ (x : α), LT.lt x a → Eq (f x) (f a)
        hb : ∀ (x : α), LT.lt b x → Eq (f x) (f b)
        x : α
        hax : LE.le a x
        hbx : LT.lt b x
        ⊢ Eq (Set.IccExtend h (Function.comp f Subtype.val) x) (f x)
      -/
    · simp [IccExtend_of_right_le _ _ hbx.le, hb x hbx]
      /-
        🎉 no goals
      -/


protected theorem Monotone.IciExtend {f : Ici a → β} (hf : Monotone f) : Monotone (IciExtend f) :=
  hf.comp monotone_projIci


protected theorem Monotone.IicExtend {f : Iic b → β} (hf : Monotone f) : Monotone (IicExtend f) :=
  hf.comp monotone_projIic


protected theorem Monotone.IccExtend (hf : Monotone f) : Monotone (IccExtend h f) :=
  hf.comp <| monotone_projIcc h


theorem StrictMono.strictMonoOn_IciExtend {f : Ici a → β} (hf : StrictMono f) :
    StrictMonoOn (IciExtend f) (Ici a) :=
  hf.comp_strictMonoOn strictMonoOn_projIci


theorem StrictMono.strictMonoOn_IicExtend {f : Iic b → β} (hf : StrictMono f) :
    StrictMonoOn (IicExtend f) (Iic b) :=
  hf.comp_strictMonoOn strictMonoOn_projIic


theorem StrictMono.strictMonoOn_IccExtend (hf : StrictMono f) :
    StrictMonoOn (IccExtend h f) (Icc a b) :=
  hf.comp_strictMonoOn (strictMonoOn_projIcc h)


protected theorem Set.OrdConnected.IciExtend {s : Set (Ici a)} (hs : s.OrdConnected) :
    {x | IciExtend (· ∈ s) x}.OrdConnected :=
  ⟨fun _ hx _ hy _ hz => hs.out hx hy ⟨max_le_max le_rfl hz.1, max_le_max le_rfl hz.2⟩⟩


protected theorem Set.OrdConnected.IicExtend {s : Set (Iic b)} (hs : s.OrdConnected) :
    {x | IicExtend (· ∈ s) x}.OrdConnected :=
  ⟨fun _ hx _ hy _ hz => hs.out hx hy ⟨min_le_min le_rfl hz.1, min_le_min le_rfl hz.2⟩⟩


protected theorem Set.OrdConnected.restrict (hs : s.OrdConnected) :
    {x | restrict t (· ∈ s) x}.OrdConnected :=
  ⟨fun _ hx _ hy _ hz => hs.out hx hy hz⟩

