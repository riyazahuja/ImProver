/-- Given a functor `F : Jᵒᵖ ⥤ Type v` where `J` is a well-ordered type, this data
allows to construct a section of `F` from an element in `F.obj (op ⊥)`,
see `WellOrderInductionData.sectionsMk`. -/
structure WellOrderInductionData where
  /-- A section `F.obj (op j) → F.obj (op (Order.succ j))` to the restriction
  `F.obj (op (Order.succ j)) → F.obj (op j)` when `j` is not maximal. -/
  succ (j : J) (hj : ¬IsMax j) (x : F.obj (op j)) : F.obj (op (Order.succ j))
  map_succ (j : J) (hj : ¬IsMax j) (x : F.obj (op j)) :
      F.map (homOfLE (Order.le_succ j)).op (succ j hj x) = x
  /-- When `j` is a limit element, and `x` is a compatible family of elements
  in `F.obj (op i)` for all `i < j`, this is a lifting to `F.obj (op j)`. -/
  lift (j : J) (hj : Order.IsSuccLimit j)
    (x : ((OrderHom.Subtype.val (· ∈ Set.Iio j)).monotone.functor.op ⋙ F).sections) :
      F.obj (op j)
  map_lift (j : J) (hj : Order.IsSuccLimit j)
    (x : ((OrderHom.Subtype.val (· ∈ Set.Iio j)).monotone.functor.op ⋙ F).sections)
    (i : J) (hi : i < j) :
        F.map (homOfLE hi.le).op (lift j hj x) = x.val (op ⟨i, hi⟩)


/-- Given `d : F.WellOrderInductionData`, `val₀ : F.obj (op ⊥)` and `j : J`,
this is the data of an element `val : F.obj (op j)` such that the induced
compatible family of elements in all `F.obj (op i)` for `i ≤ j`
is determined by `val₀` and the choice of "liftings" given by `d`. -/
structure Extension (val₀ : F.obj (op ⊥)) (j : J) where
  /-- An element in `F.obj (op j)`, which, by restriction, induces elements
  in `F.obj (op i)` for all `i ≤ j`. -/
  val : F.obj (op j)
  map_zero : F.map (homOfLE bot_le).op val = val₀
  map_succ (i : J) (hi : i < j) :
    F.map (homOfLE (Order.succ_le_of_lt hi)).op val =
      d.succ i (not_isMax_iff.2 ⟨_, hi⟩) (F.map (homOfLE hi.le).op val)
  map_limit (i : J) (hi : Order.IsSuccLimit i) (hij : i ≤ j) :
    F.map (homOfLE hij).op val = d.lift i hi
      { val := fun ⟨⟨k, hk⟩⟩ ↦ F.map (homOfLE (hk.le.trans hij)).op val
        property := fun f ↦ by
          /-
            J : Type u
            inst✝² : LinearOrder J
            inst✝¹ : SuccOrder J
            F : CategoryTheory.Functor (Opposite J) (Type v)
            d : F.WellOrderInductionData
            inst✝ : OrderBot J
            val₀ : F.obj { unop := Bot.bot }
            j : J
            val : F.obj { unop := j }
            map_zero : Eq (F.map (CategoryTheory.homOfLE ⋯).op val) val₀
            map_succ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).op …
            i : J
            hi : Order.IsSuccLimit i
            hij : LE.le i j
            j✝ j'✝ : Opposite (Subtype fun x => Membership.mem (Set.Iio i) x)
            f : Quiver.Hom j✝ j'✝
            ⊢ Eq ((⋯.functor.op.comp F).map f ((fun x => CategoryTheory.Functor.WellOrderI …
          -/
          dsimp
          /-
            J : Type u
            inst✝² : LinearOrder J
            inst✝¹ : SuccOrder J
            F : CategoryTheory.Functor (Opposite J) (Type v)
            d : F.WellOrderInductionData
            inst✝ : OrderBot J
            val₀ : F.obj { unop := Bot.bot }
            j : J
            val : F.obj { unop := j }
            map_zero : Eq (F.map (CategoryTheory.homOfLE ⋯).op val) val₀
            map_succ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).op …
            i : J
            hi : Order.IsSuccLimit i
            hij : LE.le i j
            j✝ j'✝ : Opposite (Subtype fun x => Membership.mem (Set.Iio i) x)
            f : Quiver.Hom j✝ j'✝
            ⊢ Eq (F.map (⋯.functor.map f.unop).op (F.map (CategoryTheory.homOfLE ⋯).op val …
          -/
          rw [← FunctorToTypes.map_comp_apply]
          /-
            J : Type u
            inst✝² : LinearOrder J
            inst✝¹ : SuccOrder J
            F : CategoryTheory.Functor (Opposite J) (Type v)
            d : F.WellOrderInductionData
            inst✝ : OrderBot J
            val₀ : F.obj { unop := Bot.bot }
            j : J
            val : F.obj { unop := j }
            map_zero : Eq (F.map (CategoryTheory.homOfLE ⋯).op val) val₀
            map_succ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).op …
            i : J
            hi : Order.IsSuccLimit i
            hij : LE.le i j
            j✝ j'✝ : Opposite (Subtype fun x => Membership.mem (Set.Iio i) x)
            f : Quiver.Hom j✝ j'✝
            ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE ⋯).op  …
          -/
          rfl }
          /-
            🎉 no goals
          -/


/-- An element in `d.Extension val₀ j` induces an element in `d.Extension val₀ i` when `i ≤ j`. -/
@[simps]
def ofLE {j : J} (e : d.Extension val₀ j) {i : J} (hij : i ≤ j) : d.Extension val₀ i where
  val := F.map (homOfLE hij).op e.val
  map_zero := by
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (F.map (CategoryTheory.homOfLE hij). …
    -/
    rw [← FunctorToTypes.map_comp_apply]
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE hij).o …
    -/
    exact e.map_zero
    /-
      🎉 no goals
    -/
  map_succ k hk := by
    rw [← FunctorToTypes.map_comp_apply, ← FunctorToTypes.map_comp_apply, ← op_comp, ← op_comp,
      homOfLE_comp, homOfLE_comp, e.map_succ k (lt_of_lt_of_le hk hij)]
  map_limit k hk hki := by
    rw [← FunctorToTypes.map_comp_apply, ← op_comp, homOfLE_comp,
      e.map_limit k hk (hki.trans hij)]
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      k : J
      hk : Order.IsSuccLimit k
      hki : LE.le k i
      ⊢ Eq (d.lift k hk ⟨fun x => CategoryTheory.Functor.WellOrderInductionData.Exte …
    -/
    congr
    /-
      case e_x.e_val
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      k : J
      hk : Order.IsSuccLimit k
      hki : LE.le k i
      ⊢ Eq (fun x => CategoryTheory.Functor.WellOrderInductionData.Extension.match_1 …
    -/
    ext ⟨l, hl⟩
    /-
      case e_x.e_val.h.op.mk
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      k : J
      hk : Order.IsSuccLimit k
      hki : LE.le k i
      l : J
      hl : Membership.mem (Set.Iio k) l
      ⊢ Eq (CategoryTheory.Functor.WellOrderInductionData.Extension.match_1 k (fun x …
    -/
    dsimp
    /-
      case e_x.e_val.h.op.mk
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      k : J
      hk : Order.IsSuccLimit k
      hki : LE.le k i
      l : J
      hl : Membership.mem (Set.Iio k) l
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op e.val) (F.map (CategoryTheory.homOfL …
    -/
    rw [← FunctorToTypes.map_comp_apply]
    /-
      case e_x.e_val.h.op.mk
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      i : J
      hij : LE.le i j
      k : J
      hk : Order.IsSuccLimit k
      hki : LE.le k i
      l : J
      hl : Membership.mem (Set.Iio k) l
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op e.val) (F.map (CategoryTheory.Catego …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma val_injective {j : J} {e e' : d.Extension val₀ j} (h : e.val = e'.val) : e = e' := by
  /-
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    j : J
    e e' : d.Extension val₀ j
    h : Eq e.val e'.val
    ⊢ Eq e e'
  -/
  cases e
  /-
    case mk
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    j : J
    e' : d.Extension val₀ j
    val✝ : F.obj { unop := j }
    map_zero✝ : Eq (F.map (CategoryTheory.homOfLE ⋯).op val✝) val₀
    map_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).o …
    map_limit✝ : ∀ (i : J) (hi : Order.IsSuccLimit i) (hij : LE.le i j), Eq (F.map …
    h : Eq { val := val✝, map_zero := map_zero✝, map_succ := map_succ✝, map_limit  …
    ⊢ Eq { val := val✝, map_zero := map_zero✝, map_succ := map_succ✝, map_limit := …
  -/
  cases e'
  /-
    case mk.mk
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    j : J
    val✝¹ : F.obj { unop := j }
    map_zero✝¹ : Eq (F.map (CategoryTheory.homOfLE ⋯).op val✝¹) val₀
    map_succ✝¹ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯). …
    map_limit✝¹ : ∀ (i : J) (hi : Order.IsSuccLimit i) (hij : LE.le i j), Eq (F.ma …
    val✝ : F.obj { unop := j }
    map_zero✝ : Eq (F.map (CategoryTheory.homOfLE ⋯).op val✝) val₀
    map_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).o …
    map_limit✝ : ∀ (i : J) (hi : Order.IsSuccLimit i) (hij : LE.le i j), Eq (F.map …
    h : Eq { val := val✝¹, map_zero := map_zero✝¹, map_succ := map_succ✝¹, map_lim …
    ⊢ Eq { val := val✝¹, map_zero := map_zero✝¹, map_succ := map_succ✝¹, map_limit …
  -/
  subst h
  /-
    case mk.mk
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    j : J
    val✝ : F.obj { unop := j }
    map_zero✝¹ : Eq (F.map (CategoryTheory.homOfLE ⋯).op val✝) val₀
    map_succ✝¹ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯). …
    map_limit✝¹ : ∀ (i : J) (hi : Order.IsSuccLimit i) (hij : LE.le i j), Eq (F.ma …
    map_zero✝ : Eq (F.map (CategoryTheory.homOfLE ⋯).op { val := val✝, map_zero := …
    map_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq (F.map (CategoryTheory.homOfLE ⋯).o …
    map_limit✝ : ∀ (i : J) (hi : Order.IsSuccLimit i) (hij : LE.le i j), Eq (F.map …
    ⊢ Eq { val := val✝, map_zero := map_zero✝¹, map_succ := map_succ✝¹, map_limit  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance [WellFoundedLT J] (j : J) : Subsingleton (d.Extension val₀ j) := by
  induction j using SuccOrder.limitRecOn with
  | hm i hi =>
      obtain rfl : i = ⊥ := by simpa using hi
      refine Subsingleton.intro (fun e₁ e₂ ↦ val_injective ?_)
      have h₁ := e₁.map_zero
      have h₂ := e₂.map_zero
      simp only [homOfLE_refl, op_id, FunctorToTypes.map_id_apply] at h₁ h₂
      rw [h₁, h₂]
  | hs i hi hi' =>
      refine Subsingleton.intro (fun e₁ e₂ ↦ val_injective ?_)
      have h₁ := e₁.map_succ i (Order.lt_succ_of_not_isMax hi)
      have h₂ := e₂.map_succ i (Order.lt_succ_of_not_isMax hi)
      simp only [homOfLE_refl, op_id, FunctorToTypes.map_id_apply, homOfLE_leOfHom] at h₁ h₂
      rw [h₁, h₂]
      congr
      exact congr_arg val
        (Subsingleton.elim (e₁.ofLE (Order.le_succ i)) (e₂.ofLE (Order.le_succ i)))
  | hl i hi hi' =>
      refine Subsingleton.intro (fun e₁ e₂ ↦ val_injective ?_)
      have h₁ := e₁.map_limit i hi (by rfl)
      have h₂ := e₂.map_limit i hi (by rfl)
      simp only [homOfLE_refl, op_id, FunctorToTypes.map_id_apply, OrderHom.Subtype.val_coe,
        comp_obj, op_obj, Monotone.functor_obj, homOfLE_leOfHom] at h₁ h₂
      rw [h₁, h₂]
      congr
      ext ⟨⟨l, hl⟩⟩
      have := hi' l hl
      exact congr_arg val (Subsingleton.elim (e₁.ofLE hl.le) (e₂.ofLE hl.le))


lemma compatibility [WellFoundedLT J]
    {j : J} (e : d.Extension val₀ j) {i : J} (e' : d.Extension val₀ i) (h : i ≤ j) :
    F.map (homOfLE h).op e.val = e'.val := by
  /-
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝¹ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    inst✝ : WellFoundedLT J
    j : J
    e : d.Extension val₀ j
    i : J
    e' : d.Extension val₀ i
    h : LE.le i j
    ⊢ Eq (F.map (CategoryTheory.homOfLE h).op e.val) e'.val
  -/
  obtain rfl : e' = e.ofLE h := Subsingleton.elim _ _
  /-
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝¹ : OrderBot J
    val₀ : F.obj { unop := Bot.bot }
    inst✝ : WellFoundedLT J
    j : J
    e : d.Extension val₀ j
    i : J
    h : LE.le i j
    ⊢ Eq (F.map (CategoryTheory.homOfLE h).op e.val) (e.ofLE h).val
  -/
  rfl
  /-
    🎉 no goals
  -/


variable (d val₀) in
/-- The obvious element in `d.Extension val₀ ⊥`. -/
@[simps]
def zero : d.Extension val₀ ⊥ where
  val := val₀
                 /-
                   J : Type u
                   inst✝² : LinearOrder J
                   inst✝¹ : SuccOrder J
                   F : CategoryTheory.Functor (Opposite J) (Type v)
                   d : F.WellOrderInductionData
                   inst✝ : OrderBot J
                   val₀ : F.obj { unop := Bot.bot }
                   ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op val₀) val₀
                 -/
  map_zero := by simp
                 /-
                   🎉 no goals
                 -/
                      /-
                        J : Type u
                        inst✝² : LinearOrder J
                        inst✝¹ : SuccOrder J
                        F : CategoryTheory.Functor (Opposite J) (Type v)
                        d : F.WellOrderInductionData
                        inst✝ : OrderBot J
                        val₀ : F.obj { unop := Bot.bot }
                        i : J
                        hi : LT.lt i Bot.bot
                        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op val₀) (d.succ i ⋯ (F.map (CategoryTh …
                      -/
  map_succ i hi := by simp at hi
                      /-
                        🎉 no goals
                      -/
  map_limit i hi hij := by
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i Bot.bot
      ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op val₀) (d.lift i hi ⟨fun x => Categ …
    -/
    obtain rfl : i = ⊥ := by simpa using hij
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      hi : Order.IsSuccLimit Bot.bot
      hij : LE.le Bot.bot Bot.bot
      ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op val₀) (d.lift Bot.bot hi ⟨fun x => …
    -/
    simpa using hi.not_isMin
    /-
      🎉 no goals
    -/


/-- The element in `d.Extension val₀ (Order.succ j)` obtained by extending
an element in `d.Extension val₀ j` when `j` is not maximal. -/
def succ {j : J} (e : d.Extension val₀ j) (hj : ¬IsMax j) :
    d.Extension val₀ (Order.succ j) where
  val := d.succ j hj e.val
  map_zero := by
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) val₀
    -/
    simp only [← e.map_zero]
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (F.map (Categor …
    -/
    conv_rhs => rw [← d.map_succ j hj e.val]
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (F.map (Categor …
    -/
    rw [← FunctorToTypes.map_comp_apply]
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (F.map (Categor …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_succ i hi := by
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      i : J
      hi : LT.lt i (Order.succ j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.succ i ⋯ (F. …
    -/
    obtain hij | rfl := ((Order.lt_succ_iff_of_not_isMax hj).mp hi).lt_or_eq
    · rw [← homOfLE_comp ((Order.lt_succ_iff_of_not_isMax hj).mp hi) (Order.le_succ j), op_comp,
        FunctorToTypes.map_comp_apply, d.map_succ, ← e.map_succ i hij,
        ← homOfLE_comp (Order.succ_le_of_lt hij) (Order.le_succ j), op_comp,
        FunctorToTypes.map_comp_apply, d.map_succ]
    · simp only [homOfLE_refl, op_id, FunctorToTypes.map_id_apply, homOfLE_leOfHom,
        d.map_succ]
  map_limit i hi hij := by
    /-
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      j : J
      e : d.Extension val₀ j
      hj : Not (IsMax j)
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i (Order.succ j)
      ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.succ j hj e.val)) (d.lift i hi  …
    -/
    obtain hij | rfl := hij.lt_or_eq
      /-
        case inl
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij✝).op (d.succ j hj e.val)) (d.lift i hi …
      -/
    · have hij' : i ≤ j := (Order.lt_succ_iff_of_not_isMax hj).mp hij
      /-
        case inl
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij✝).op (d.succ j hj e.val)) (d.lift i hi …
      -/
      have := congr_arg (F.map (homOfLE hij').op) (d.map_succ j hj e.val)
      /-
        case inl
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE hij').op (F.map (CategoryTheory.homOf …
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij✝).op (d.succ j hj e.val)) (d.lift i hi …
      -/
      rw [e.map_limit i hi, ← FunctorToTypes.map_comp_apply, ← op_comp, homOfLE_comp] at this
      /-
        case inl
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij✝).op (d.succ j hj e.val)) (d.lift i hi …
      -/
      rw [this]
      /-
        case inl
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        ⊢ Eq (d.lift i hi ⟨fun x => CategoryTheory.Functor.WellOrderInductionData.Exte …
      -/
      congr
      /-
        case inl.e_x.e_val
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        ⊢ Eq (fun x => CategoryTheory.Functor.WellOrderInductionData.Extension.match_1 …
      -/
      ext ⟨⟨l, hl⟩⟩
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (CategoryTheory.Functor.WellOrderInductionData.Extension.match_1 i (fun x …
      -/
      dsimp
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op e.val) (F.map (CategoryTheory.homOfL …
      -/
      conv_lhs => rw [← d.map_succ j hj e.val]
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (F.map (CategoryTheory.homOfLE ⋯).op …
      -/
      rw [← FunctorToTypes.map_comp_apply]
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        i : J
        hi : Order.IsSuccLimit i
        hij✝ : LE.le i (Order.succ j)
        hij : LT.lt i (Order.succ j)
        hij' : LE.le i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.succ j hj e.val)) (d.lift i  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE ⋯).op  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case inr
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        hi : Order.IsSuccLimit (Order.succ j)
        hij : LE.le (Order.succ j) (Order.succ j)
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.succ j hj e.val)) (d.lift (Orde …
      -/
    · exfalso
      /-
        case inr
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        j : J
        e : d.Extension val₀ j
        hj : Not (IsMax j)
        hi : Order.IsSuccLimit (Order.succ j)
        hij : LE.le (Order.succ j) (Order.succ j)
        ⊢ False
      -/
      exact hj hi.isMax
      /-
        🎉 no goals
      -/


/-- When `j` is a limit element, this is the exntesion to `d.Extension val₀ j`
of a family of elements in `d.Extension val₀ i` for all `i < j`. -/
def limit (j : J) (hj : Order.IsSuccLimit j)
    (e : ∀ (i : J) (_ : i < j), d.Extension val₀ i) :
    d.Extension val₀ j where
  val := d.lift j hj
    { val := fun ⟨i, hi⟩ ↦ (e i hi).val
                             /-
                               J : Type u
                               inst✝³ : LinearOrder J
                               inst✝² : SuccOrder J
                               F : CategoryTheory.Functor (Opposite J) (Type v)
                               d : F.WellOrderInductionData
                               inst✝¹ : OrderBot J
                               val₀ : F.obj { unop := Bot.bot }
                               inst✝ : WellFoundedLT J
                               j : J
                               hj : Order.IsSuccLimit j
                               e : (i : J) → LT.lt i j → d.Extension val₀ i
                               j✝ j'✝ : Opposite (Subtype fun x => Membership.mem (Set.Iio j) x)
                               f : Quiver.Hom j✝ j'✝
                               ⊢ Eq ((⋯.functor.op.comp F).map f ((fun x => CategoryTheory.Functor.WellOrderI …
                             -/
      property := fun f ↦ by apply compatibility }
                             /-
                               🎉 no goals
                             -/
  map_zero := by
    /-
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝¹ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      inst✝ : WellFoundedLT J
      j : J
      hj : Order.IsSuccLimit j
      e : (i : J) → LT.lt i j → d.Extension val₀ i
      ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.lift j hj ⟨fun x => CategoryTheor …
    -/
    rw [d.map_lift _ _ _ _ (by simpa [bot_lt_iff_ne_bot] using hj.not_isMin)]
    simpa only [homOfLE_refl, op_id, FunctorToTypes.map_id_apply] using
      (e ⊥ (by simpa [bot_lt_iff_ne_bot] using hj.not_isMin)).map_zero
  map_succ i hi := by
    convert (e (Order.succ i) ((Order.IsSuccLimit.succ_lt_iff hj).mpr hi)).map_succ i
      (by
        simp only [Order.lt_succ_iff_not_isMax, not_isMax_iff]
        exact ⟨_, hi⟩) using 1
      /-
        case h.e'_2
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : LT.lt i j
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.lift j hj ⟨fun x => CategoryTheor …
      -/
    · dsimp
      rw [FunctorToTypes.map_id_apply,
        d.map_lift _ _ _ _ ((Order.IsSuccLimit.succ_lt_iff hj).mpr hi)]
      /-
        case h.e'_3
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : LT.lt i j
        ⊢ Eq (d.succ i ⋯ (F.map (CategoryTheory.homOfLE ⋯).op (d.lift j hj ⟨fun x => C …
      -/
    · congr
      /-
        case h.e'_3.e_x
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : LT.lt i j
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (d.lift j hj ⟨fun x => CategoryTheor …
      -/
      rw [d.map_lift _ _ _ _ hi]
      /-
        case h.e'_3.e_x
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : LT.lt i j
        ⊢ Eq (↑⟨fun x => CategoryTheory.Functor.WellOrderInductionData.Extension.match …
      -/
      symm
      /-
        case h.e'_3.e_x
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : LT.lt i j
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (e (Order.succ i) ⋯).val) (↑⟨fun x = …
      -/
      apply compatibility
      /-
        🎉 no goals
      -/
  map_limit i hi hij := by
    /-
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : SuccOrder J
      F : CategoryTheory.Functor (Opposite J) (Type v)
      d : F.WellOrderInductionData
      inst✝¹ : OrderBot J
      val₀ : F.obj { unop := Bot.bot }
      inst✝ : WellFoundedLT J
      j : J
      hj : Order.IsSuccLimit j
      e : (i : J) → LT.lt i j → d.Extension val₀ i
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i j
      ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift j hj ⟨fun x => CategoryThe …
    -/
    obtain hij' | rfl := hij.lt_or_eq
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift j hj ⟨fun x => CategoryThe …
      -/
    · have := (e i hij').map_limit i hi (by rfl)
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val) (d.lift i hi ⟨f …
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift j hj ⟨fun x => CategoryThe …
      -/
      dsimp at this ⊢
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (F.map (CategoryTheory.CategoryStruct.id { unop := i }) (e i hij').v …
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift j hj ⟨fun x => (e ↑x.1 ⋯). …
      -/
      rw [FunctorToTypes.map_id_apply] at this
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift j hj ⟨fun x => (e ↑x.1 ⋯). …
      -/
      rw [d.map_lift _ _ _ _ hij']
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (↑⟨fun x => (e ↑x.1 ⋯).val, ⋯⟩ { unop := ⟨i, hij'⟩ }) (d.lift i hi ⟨fun x …
      -/
      dsimp
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (e i ⋯).val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE ⋯).op (d …
      -/
      rw [this]
      /-
        case inl
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val …
      -/
      congr
      /-
        case inl.e_x.e_val
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (fun x => F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val) fun x => F. …
      -/
      dsimp
      /-
        case inl.e_x.e_val
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        ⊢ Eq (fun x => F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val) fun x => F. …
      -/
      ext ⟨⟨l, hl⟩⟩
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val) (F.map (CategoryTheo …
      -/
      rw [map_lift _ _ _ _ _ (hl.trans hij')]
      /-
        case inl.e_x.e_val.h.op.mk
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        j : J
        hj : Order.IsSuccLimit j
        e : (i : J) → LT.lt i j → d.Extension val₀ i
        i : J
        hi : Order.IsSuccLimit i
        hij : LE.le i j
        hij' : LT.lt i j
        this : Eq (e i hij').val (d.lift i hi ⟨fun x => F.map (CategoryTheory.homOfLE  …
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯).op (e i hij').val) (↑⟨fun x => (e ↑x.1  …
      -/
      apply compatibility
      /-
        🎉 no goals
      -/
      /-
        case inr
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        i : J
        hi hj : Order.IsSuccLimit i
        e : (i_1 : J) → LT.lt i_1 i → d.Extension val₀ i_1
        hij : LE.le i i
        ⊢ Eq (F.map (CategoryTheory.homOfLE hij).op (d.lift i hj ⟨fun x => CategoryThe …
      -/
    · dsimp
      /-
        case inr
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        i : J
        hi hj : Order.IsSuccLimit i
        e : (i_1 : J) → LT.lt i_1 i → d.Extension val₀ i_1
        hij : LE.le i i
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id { unop := i }) (d.lift i hj ⟨fun …
      -/
      rw [FunctorToTypes.map_id_apply]
      /-
        case inr
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        i : J
        hi hj : Order.IsSuccLimit i
        e : (i_1 : J) → LT.lt i_1 i → d.Extension val₀ i_1
        hij : LE.le i i
        ⊢ Eq (d.lift i hj ⟨fun x => (e ↑x.1 ⋯).val, ⋯⟩) (d.lift i hi ⟨fun x => F.map ( …
      -/
      congr
      /-
        case inr.e_x.e_val
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        i : J
        hi hj : Order.IsSuccLimit i
        e : (i_1 : J) → LT.lt i_1 i → d.Extension val₀ i_1
        hij : LE.le i i
        ⊢ Eq (fun x => (e ↑x.1 ⋯).val) fun x => F.map (CategoryTheory.homOfLE ⋯).op (d …
      -/
      ext ⟨⟨l, hl⟩⟩
      /-
        case inr.e_x.e_val.h.op.mk
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : SuccOrder J
        F : CategoryTheory.Functor (Opposite J) (Type v)
        d : F.WellOrderInductionData
        inst✝¹ : OrderBot J
        val₀ : F.obj { unop := Bot.bot }
        inst✝ : WellFoundedLT J
        i : J
        hi hj : Order.IsSuccLimit i
        e : (i_1 : J) → LT.lt i_1 i → d.Extension val₀ i_1
        hij : LE.le i i
        l : J
        hl : Membership.mem (Set.Iio i) l
        ⊢ Eq (e ↑{ unop := ⟨l, hl⟩ }.1 ⋯).val (F.map (CategoryTheory.homOfLE ⋯).op (d. …
      -/
      rw [d.map_lift _ _ _ _ hl]
      /-
        🎉 no goals
      -/


instance (j : J) : Nonempty (d.Extension val₀ j) := by
  induction j using SuccOrder.limitRecOn with
  | hm i hi =>
      obtain rfl : i = ⊥ := by simpa using hi
      exact ⟨zero d val₀⟩
  | hs i hi hi' => exact ⟨hi'.some.succ hi⟩
  | hl i hi hi' => exact ⟨limit i hi (fun l hl ↦ (hi' l hl).some)⟩


noncomputable instance (j : J) : Unique (d.Extension val₀ j) :=
  uniqueOfSubsingleton (Nonempty.some inferInstance)


/-- When `J` is a well-ordered type, `F : Jᵒᵖ ⥤ Type v`, and `d : F.WellOrderInductionData`,
this is the section of `F` that is determined by `val₀ : F.obj (op ⊥)` -/
noncomputable def sectionsMk (val₀ : F.obj (op ⊥)) : F.sections where
  val j := (default : d.Extension val₀ j.unop).val
                         /-
                           J : Type u
                           inst✝³ : LinearOrder J
                           inst✝² : SuccOrder J
                           F : CategoryTheory.Functor (Opposite J) (Type v)
                           d : F.WellOrderInductionData
                           inst✝¹ : OrderBot J
                           inst✝ : WellFoundedLT J
                           val₀ : F.obj { unop := Bot.bot }
                           j✝ j'✝ : Opposite J
                           f : Quiver.Hom j✝ j'✝
                           ⊢ Eq (F.map f ((fun j => Inhabited.default.val) j✝)) ((fun j => Inhabited.defa …
                         -/
  property := fun f ↦ by apply Extension.compatibility
                         /-
                           🎉 no goals
                         -/


lemma sectionsMk_val_op_bot (val₀ : F.obj (op ⊥)) :
    (d.sectionsMk val₀).val (op ⊥) = val₀ := by
  /-
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : SuccOrder J
    F : CategoryTheory.Functor (Opposite J) (Type v)
    d : F.WellOrderInductionData
    inst✝¹ : OrderBot J
    inst✝ : WellFoundedLT J
    val₀ : F.obj { unop := Bot.bot }
    ⊢ Eq (↑(d.sectionsMk val₀) { unop := Bot.bot }) val₀
  -/
  simpa using (default : d.Extension val₀ ⊥).map_zero
  /-
    🎉 no goals
  -/


include d in
lemma surjective :
    Function.Surjective ((fun s ↦ s (op ⊥)) ∘ Subtype.val : F.sections → F.obj (op ⊥)) :=
  fun val₀ ↦ ⟨d.sectionsMk val₀, d.sectionsMk_val_op_bot val₀⟩


