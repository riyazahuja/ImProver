instance semilatticeInf [SemilatticeInf α] {a b : α} : SemilatticeInf (Ico a b) :=
  Subtype.semilatticeInf fun _ _ hx hy => ⟨le_inf hx.1 hy.1, lt_of_le_of_lt inf_le_left hx.2⟩


/-- `Ico a b` has a bottom element whenever `a < b`. -/
protected abbrev orderBot [PartialOrder α] {a b : α} (h : a < b) : OrderBot (Ico a b) :=
  (isLeast_Ico h).orderBot


instance semilatticeInf [SemilatticeInf α] {a : α} : SemilatticeInf (Iio a) :=
  Subtype.semilatticeInf fun _ _ hx _ => lt_of_le_of_lt inf_le_left hx


instance semilatticeSup [SemilatticeSup α] {a b : α} : SemilatticeSup (Ioc a b) :=
  Subtype.semilatticeSup fun _ _ hx hy => ⟨lt_of_lt_of_le hx.1 le_sup_left, sup_le hx.2 hy.2⟩


/-- `Ioc a b` has a top element whenever `a < b`. -/
protected abbrev orderTop [PartialOrder α] {a b : α} (h : a < b) : OrderTop (Ioc a b) :=
  (isGreatest_Ioc h).orderTop


instance semilatticeSup [SemilatticeSup α] {a : α} : SemilatticeSup (Ioi a) :=
  Subtype.semilatticeSup fun _ _ hx _ => lt_of_lt_of_le hx le_sup_left


instance semilatticeInf [SemilatticeInf α] : SemilatticeInf (Iic a) :=
  Subtype.semilatticeInf fun _ _ hx _ => le_trans inf_le_left hx


@[simp, norm_cast]
protected lemma coe_inf [SemilatticeInf α] {x y : Iic a} :
    (↑(x ⊓ y) : α) = (x : α) ⊓ (y : α) :=
  rfl


instance semilatticeSup [SemilatticeSup α] : SemilatticeSup (Iic a) :=
  Subtype.semilatticeSup fun _ _ hx hy => sup_le hx hy


@[simp, norm_cast]
protected lemma coe_sup [SemilatticeSup α] {x y : Iic a} :
    (↑(x ⊔ y) : α) = (x : α) ⊔ (y : α) :=
  rfl


instance [Lattice α] : Lattice (Iic a) :=
  { Iic.semilatticeInf, Iic.semilatticeSup with }


instance orderTop [Preorder α] :
    OrderTop (Iic a) where
  top := ⟨a, le_refl a⟩
  le_top x := x.prop


@[simp]
theorem coe_top [Preorder α] : (⊤ : Iic a) = a :=
  rfl


protected lemma eq_top_iff [Preorder α] {x : Iic a} :
    x = ⊤ ↔ (x : α) = a := by
  /-
    α : Type u_1
    a : α
    inst✝ : Preorder α
    x : ↑(Set.Iic a)
    ⊢ Iff (Eq x Top.top) (Eq (↑x) a)
  -/
  simp [Subtype.ext_iff]
  /-
    🎉 no goals
  -/


instance orderBot [Preorder α] [OrderBot α] :
    OrderBot (Iic a) where
  bot := ⟨⊥, bot_le⟩
  bot_le := fun ⟨_, _⟩ => Subtype.mk_le_mk.2 bot_le


@[simp]
theorem coe_bot [Preorder α] [OrderBot α] : (⊥ : Iic a) = (⊥ : α) :=
  rfl


instance [Preorder α] [OrderBot α] : BoundedOrder (Iic a) :=
  { Iic.orderTop, Iic.orderBot with }


protected lemma disjoint_iff [SemilatticeInf α] [OrderBot α] {x y : Iic a} :
    Disjoint x y ↔ Disjoint (x : α) (y : α) := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    x y : ↑(Set.Iic a)
    ⊢ Iff (Disjoint x y) (Disjoint ↑x ↑y)
  -/
  simp [_root_.disjoint_iff, Subtype.ext_iff]
  /-
    🎉 no goals
  -/


protected lemma codisjoint_iff [SemilatticeSup α] {x y : Iic a} :
    Codisjoint x y ↔ (x : α) ⊔ (y : α) = a := by
  /-
    α : Type u_1
    a : α
    inst✝ : SemilatticeSup α
    x y : ↑(Set.Iic a)
    ⊢ Iff (Codisjoint x y) (Eq (Max.max ↑x ↑y) a)
  -/
  simpa only [_root_.codisjoint_iff] using Iic.eq_top_iff
  /-
    🎉 no goals
  -/


protected lemma isCompl_iff [Lattice α] [BoundedOrder α] {x y : Iic a} :
    IsCompl x y ↔ Disjoint (x : α) (y : α) ∧ (x : α) ⊔ (y : α) = a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Lattice α
    inst✝ : BoundedOrder α
    x y : ↑(Set.Iic a)
    ⊢ Iff (IsCompl x y) (And (Disjoint ↑x ↑y) (Eq (Max.max ↑x ↑y) a))
  -/
  rw [_root_.isCompl_iff, Iic.disjoint_iff, Iic.codisjoint_iff]
  /-
    🎉 no goals
  -/


protected lemma complementedLattice_iff [Lattice α] [BoundedOrder α] :
    ComplementedLattice (Iic a) ↔ ∀ b, b ≤ a → ∃ c ≤ a, b ⊓ c = ⊥ ∧ b ⊔ c = a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Lattice α
    inst✝ : BoundedOrder α
    ⊢ Iff (ComplementedLattice ↑(Set.Iic a)) (∀ (b : α), LE.le b a → Exists fun c  …
  -/
  refine ⟨fun h b hb ↦ ?_, fun h ↦ ⟨fun ⟨x, hx⟩ ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ComplementedLattice ↑(Set.Iic a)
      b : α
      hb : LE.le b a
      ⊢ Exists fun c => And (LE.le c a) (And (Eq (Min.min b c) Bot.bot) (Eq (Max.max …
    -/
  · obtain ⟨⟨c, hc₁⟩, hc⟩ := exists_isCompl (⟨b, hb⟩ : Iic a)
    /-
      case refine_1.intro.mk
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ComplementedLattice ↑(Set.Iic a)
      b : α
      hb : LE.le b a
      c : α
      hc₁ : Membership.mem (Set.Iic a) c
      hc : IsCompl ⟨b, hb⟩ ⟨c, hc₁⟩
      ⊢ Exists fun c => And (LE.le c a) (And (Eq (Min.min b c) Bot.bot) (Eq (Max.max …
    -/
    obtain ⟨hc₂, hc₃⟩ := Set.Iic.isCompl_iff.mp hc
    /-
      case refine_1.intro.mk.intro
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ComplementedLattice ↑(Set.Iic a)
      b : α
      hb : LE.le b a
      c : α
      hc₁ : Membership.mem (Set.Iic a) c
      hc : IsCompl ⟨b, hb⟩ ⟨c, hc₁⟩
      hc₂ : Disjoint ↑⟨b, hb⟩ ↑⟨c, hc₁⟩
      hc₃ : Eq (Max.max ↑⟨b, hb⟩ ↑⟨c, hc₁⟩) a
      ⊢ Exists fun c => And (LE.le c a) (And (Eq (Min.min b c) Bot.bot) (Eq (Max.max …
    -/
    exact ⟨c, hc₁, disjoint_iff.mp hc₂, hc₃⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ∀ (b : α), LE.le b a → Exists fun c => And (LE.le c a) (And (Eq (Min.min b …
      x✝ : ↑(Set.Iic a)
      x : α
      hx : Membership.mem (Set.Iic a) x
      ⊢ Exists fun b => IsCompl ⟨x, hx⟩ b
    -/
  · simp_rw [Set.Iic.isCompl_iff]
    /-
      case refine_2
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ∀ (b : α), LE.le b a → Exists fun c => And (LE.le c a) (And (Eq (Min.min b …
      x✝ : ↑(Set.Iic a)
      x : α
      hx : Membership.mem (Set.Iic a) x
      ⊢ Exists fun b => And (Disjoint x ↑b) (Eq (Max.max x ↑b) a)
    -/
    obtain ⟨c, hc₁, hc₂, hc₃⟩ := h x hx
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      a : α
      inst✝¹ : Lattice α
      inst✝ : BoundedOrder α
      h : ∀ (b : α), LE.le b a → Exists fun c => And (LE.le c a) (And (Eq (Min.min b …
      x✝ : ↑(Set.Iic a)
      x : α
      hx : Membership.mem (Set.Iic a) x
      c : α
      hc₁ : LE.le c a
      hc₂ : Eq (Min.min x c) Bot.bot
      hc₃ : Eq (Max.max x c) a
      ⊢ Exists fun b => And (Disjoint x ↑b) (Eq (Max.max x ↑b) a)
    -/
    exact ⟨⟨c, hc₁⟩, disjoint_iff.mpr hc₂, hc₃⟩
    /-
      🎉 no goals
    -/


instance semilatticeInf [SemilatticeInf α] {a : α} : SemilatticeInf (Ici a) :=
  Subtype.semilatticeInf fun _ _ hx hy => le_inf hx hy


instance semilatticeSup [SemilatticeSup α] {a : α} : SemilatticeSup (Ici a) :=
  Subtype.semilatticeSup fun _ _ hx _ => le_trans hx le_sup_left


instance lattice [Lattice α] {a : α} : Lattice (Ici a) :=
  { Ici.semilatticeInf, Ici.semilatticeSup with }


instance distribLattice [DistribLattice α] {a : α} : DistribLattice (Ici a) :=
  { Ici.lattice with le_sup_inf := fun _ _ _ => le_sup_inf }


instance orderBot [Preorder α] {a : α} :
    OrderBot (Ici a) where
  bot := ⟨a, le_refl a⟩
  bot_le x := x.prop


@[simp]
theorem coe_bot [Preorder α] {a : α} : ↑(⊥ : Ici a) = a :=
  rfl


instance orderTop [Preorder α] [OrderTop α] {a : α} :
    OrderTop (Ici a) where
  top := ⟨⊤, le_top⟩
  le_top := fun ⟨_, _⟩ => Subtype.mk_le_mk.2 le_top


@[simp]
theorem coe_top [Preorder α] [OrderTop α] {a : α} : ↑(⊤ : Ici a) = (⊤ : α) :=
  rfl


instance boundedOrder [Preorder α] [OrderTop α] {a : α} : BoundedOrder (Ici a) :=
  { Ici.orderTop, Ici.orderBot with }


instance semilatticeInf [SemilatticeInf α] : SemilatticeInf (Icc a b) :=
  Subtype.semilatticeInf fun _ _ hx hy => ⟨le_inf hx.1 hy.1, le_trans inf_le_left hx.2⟩


instance semilatticeSup [SemilatticeSup α] : SemilatticeSup (Icc a b) :=
  Subtype.semilatticeSup fun _ _ hx hy => ⟨le_trans hx.1 le_sup_left, sup_le hx.2 hy.2⟩


instance lattice [Lattice α] : Lattice (Icc a b) :=
  { Icc.semilatticeInf, Icc.semilatticeSup with }


/-- `Icc a b` has a bottom element whenever `a ≤ b`. -/
instance : OrderBot (Icc a b) :=
  (isLeast_Icc Fact.out).orderBot


/-- `Icc a b` has a top element whenever `a ≤ b`. -/
instance : OrderTop (Icc a b) :=
  (isGreatest_Icc Fact.out).orderTop


/-- `Icc a b` is a `BoundedOrder` whenever `a ≤ b`. -/
instance : BoundedOrder (Icc a b) where


