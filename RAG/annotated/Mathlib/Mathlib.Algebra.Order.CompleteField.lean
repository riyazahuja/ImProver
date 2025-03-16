/-- A field which is both linearly ordered and conditionally complete with respect to the order.
This axiomatizes the reals. -/
-- @[protect_proj] -- Porting note: does not exist anymore
class ConditionallyCompleteLinearOrderedField (α : Type*) extends
    LinearOrderedField α, ConditionallyCompleteLinearOrder α

-- see Note [lower instance priority]

/-- Any conditionally complete linearly ordered field is archimedean. -/
instance (priority := 100) ConditionallyCompleteLinearOrderedField.to_archimedean
    [ConditionallyCompleteLinearOrderedField α] : Archimedean α :=
  archimedean_iff_nat_lt.2
    (by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : ConditionallyCompleteLinearOrderedField α
        ⊢ ∀ (x : α), Exists fun n => LT.lt x ↑n
      -/
      by_contra! h
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : ConditionallyCompleteLinearOrderedField α
        h : Exists fun x => ∀ (n : Nat), LE.le (↑n) x
        ⊢ False
      -/
      obtain ⟨x, h⟩ := h
      have := csSup_le _ _ (range_nonempty Nat.cast)
        (forall_mem_range.2 fun m =>
          le_sub_iff_add_le.2 <| le_csSup _ _ ⟨x, forall_mem_range.2 h⟩ ⟨m+1, Nat.cast_succ m⟩)
      /-
        case intro
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝ : ConditionallyCompleteLinearOrderedField α
        x : α
        h : ∀ (n : Nat), LE.le (↑n) x
        this : LE.le (SupSet.sSup (Set.range Nat.cast)) (HSub.hSub (SupSet.sSup (Set.r …
        ⊢ False
      -/
      linarith)
      /-
        🎉 no goals
      -/


/-- The lower cut of rationals inside a linear ordered field that are less than a given element of
another linear ordered field. -/
def cutMap (a : α) : Set β :=
  (Rat.cast : ℚ → β) '' {t | ↑t < a}


theorem cutMap_mono (h : a₁ ≤ a₂) : cutMap β a₁ ⊆ cutMap β a₂ := image_subset _ fun _ => h.trans_lt'


@[simp]
theorem mem_cutMap_iff : b ∈ cutMap β a ↔ ∃ q : ℚ, (q : α) < a ∧ (q : β) = b := Iff.rfl

-- @[simp] -- Porting note: not in simpNF

theorem coe_mem_cutMap_iff [CharZero β] : (q : β) ∈ cutMap β a ↔ (q : α) < a :=
  Rat.cast_injective.mem_set_image


theorem cutMap_self (a : α) : cutMap α a = Iio a ∩ range (Rat.cast : ℚ → α) := by
  /-
    α : Type u_2
    inst✝ : LinearOrderedField α
    a : α
    ⊢ Eq (LinearOrderedField.cutMap α a) (Inter.inter (Set.Iio a) (Set.range Rat.c …
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝ : LinearOrderedField α
    a x✝ : α
    ⊢ Iff (Membership.mem (LinearOrderedField.cutMap α a) x✝) (Membership.mem (Int …
  -/
  constructor
    /-
      case h.mp
      α : Type u_2
      inst✝ : LinearOrderedField α
      a x✝ : α
      ⊢ Membership.mem (LinearOrderedField.cutMap α a) x✝ → Membership.mem (Inter.in …
    -/
  · rintro ⟨q, h, rfl⟩
    /-
      case h.mp.intro.intro
      α : Type u_2
      inst✝ : LinearOrderedField α
      a : α
      q : Rat
      h : Membership.mem (setOf fun t => LT.lt (↑t) a) q
      ⊢ Membership.mem (Inter.inter (Set.Iio a) (Set.range Rat.cast)) ↑q
    -/
    exact ⟨h, q, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_2
      inst✝ : LinearOrderedField α
      a x✝ : α
      ⊢ Membership.mem (Inter.inter (Set.Iio a) (Set.range Rat.cast)) x✝ → Membershi …
    -/
  · rintro ⟨h, q, rfl⟩
    /-
      case h.mpr.intro.intro
      α : Type u_2
      inst✝ : LinearOrderedField α
      a : α
      q : Rat
      h : Membership.mem (Set.Iio a) ↑q
      ⊢ Membership.mem (LinearOrderedField.cutMap α a) ↑q
    -/
    exact ⟨q, h, rfl⟩
    /-
      🎉 no goals
    -/


theorem cutMap_coe (q : ℚ) : cutMap β (q : α) = Rat.cast '' {r : ℚ | (r : β) < q} := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : LinearOrderedField α
    inst✝ : LinearOrderedField β
    q : Rat
    ⊢ Eq (LinearOrderedField.cutMap β ↑q) (Set.image Rat.cast (setOf fun r => LT.l …
  -/
  simp_rw [cutMap, Rat.cast_lt]
  /-
    🎉 no goals
  -/


theorem cutMap_nonempty (a : α) : (cutMap β a).Nonempty :=
  Nonempty.image _ <| exists_rat_lt a


theorem cutMap_bddAbove (a : α) : BddAbove (cutMap β a) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : LinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ⊢ BddAbove (LinearOrderedField.cutMap β a)
  -/
  obtain ⟨q, hq⟩ := exists_rat_gt a
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : LinearOrderedField β
    inst✝ : Archimedean α
    a : α
    q : Rat
    hq : LT.lt a ↑q
    ⊢ BddAbove (LinearOrderedField.cutMap β a)
  -/
  exact ⟨q, forall_mem_image.2 fun r hr => mod_cast (hq.trans' hr).le⟩
  /-
    🎉 no goals
  -/


theorem cutMap_add (a b : α) : cutMap β (a + b) = cutMap β a + cutMap β b := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : LinearOrderedField β
    inst✝ : Archimedean α
    a b : α
    ⊢ Eq (LinearOrderedField.cutMap β (HAdd.hAdd a b)) (HAdd.hAdd (LinearOrderedFi …
  -/
  refine (image_subset_iff.2 fun q hq => ?_).antisymm ?_
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : Membership.mem (setOf fun t => LT.lt (↑t) (HAdd.hAdd a b)) q
      ⊢ Membership.mem (Set.preimage Rat.cast (HAdd.hAdd (LinearOrderedField.cutMap  …
    -/
  · rw [mem_setOf_eq, ← sub_lt_iff_lt_add] at hq
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : LT.lt (HSub.hSub (↑q) b) a
      ⊢ Membership.mem (Set.preimage Rat.cast (HAdd.hAdd (LinearOrderedField.cutMap  …
    -/
    obtain ⟨q₁, hq₁q, hq₁ab⟩ := exists_rat_btwn hq
    /-
      case refine_1.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : LT.lt (HSub.hSub (↑q) b) a
      q₁ : Rat
      hq₁q : LT.lt (HSub.hSub (↑q) b) ↑q₁
      hq₁ab : LT.lt (↑q₁) a
      ⊢ Membership.mem (Set.preimage Rat.cast (HAdd.hAdd (LinearOrderedField.cutMap  …
    -/
    refine ⟨q₁, by rwa [coe_mem_cutMap_iff], q - q₁, ?_, add_sub_cancel _ _⟩
    /-
      case refine_1.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : LT.lt (HSub.hSub (↑q) b) a
      q₁ : Rat
      hq₁q : LT.lt (HSub.hSub (↑q) b) ↑q₁
      hq₁ab : LT.lt (↑q₁) a
      ⊢ Membership.mem (LinearOrderedField.cutMap β b) (HSub.hSub ↑q ↑q₁)
    -/
    norm_cast
    /-
      case refine_1.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : LT.lt (HSub.hSub (↑q) b) a
      q₁ : Rat
      hq₁q : LT.lt (HSub.hSub (↑q) b) ↑q₁
      hq₁ab : LT.lt (↑q₁) a
      ⊢ Membership.mem (LinearOrderedField.cutMap β b) ↑(HSub.hSub q q₁)
    -/
    rw [coe_mem_cutMap_iff]
    /-
      case refine_1.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      q : Rat
      hq : LT.lt (HSub.hSub (↑q) b) a
      q₁ : Rat
      hq₁q : LT.lt (HSub.hSub (↑q) b) ↑q₁
      hq₁ab : LT.lt (↑q₁) a
      ⊢ LT.lt (↑(HSub.hSub q q₁)) b
    -/
    exact mod_cast sub_lt_comm.mp hq₁q
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      ⊢ HasSubset.Subset (HAdd.hAdd (LinearOrderedField.cutMap β a) (LinearOrderedFi …
    -/
  · rintro _ ⟨_, ⟨qa, ha, rfl⟩, _, ⟨qb, hb, rfl⟩, rfl⟩
    -- After https://github.com/leanprover/lean4/pull/2734, `norm_cast` needs help with beta reduction.
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      qa : Rat
      ha : Membership.mem (setOf fun t => LT.lt (↑t) a) qa
      qb : Rat
      hb : Membership.mem (setOf fun t => LT.lt (↑t) b) qb
      ⊢ Membership.mem (Set.image Rat.cast (setOf fun t => LT.lt (↑t) (HAdd.hAdd a b …
    -/
    refine ⟨qa + qb, ?_, by beta_reduce; norm_cast⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      qa : Rat
      ha : Membership.mem (setOf fun t => LT.lt (↑t) a) qa
      qb : Rat
      hb : Membership.mem (setOf fun t => LT.lt (↑t) b) qb
      ⊢ Membership.mem (setOf fun t => LT.lt (↑t) (HAdd.hAdd a b)) (HAdd.hAdd qa qb)
    -/
    rw [mem_setOf_eq, cast_add]
    /-
      case refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : LinearOrderedField β
      inst✝ : Archimedean α
      a b : α
      qa : Rat
      ha : Membership.mem (setOf fun t => LT.lt (↑t) a) qa
      qb : Rat
      hb : Membership.mem (setOf fun t => LT.lt (↑t) b) qb
      ⊢ LT.lt (HAdd.hAdd ↑qa ↑qb) (HAdd.hAdd a b)
    -/
    exact add_lt_add ha hb
    /-
      🎉 no goals
    -/


/-- The induced order preserving function from a linear ordered field to a conditionally complete
linear ordered field, defined by taking the Sup in the codomain of all the rationals less than the
input. -/
def inducedMap (x : α) : β :=
  sSup <| cutMap β x


theorem inducedMap_mono : Monotone (inducedMap α β) := fun _ _ h =>
  csSup_le_csSup (cutMap_bddAbove β _) (cutMap_nonempty β _) (cutMap_mono β h)


theorem inducedMap_rat (q : ℚ) : inducedMap α β (q : α) = q := by
  refine csSup_eq_of_forall_le_of_forall_lt_exists_gt
    (cutMap_nonempty β (q : α)) (fun x h => ?_) fun w h => ?_
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q : Rat
      x : β
      h : Membership.mem (LinearOrderedField.cutMap β ↑q) x
      ⊢ LE.le x ↑q
    -/
  · rw [cutMap_coe] at h
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q : Rat
      x : β
      h : Membership.mem (Set.image Rat.cast (setOf fun r => LT.lt ↑r ↑q)) x
      ⊢ LE.le x ↑q
    -/
    obtain ⟨r, h, rfl⟩ := h
    /-
      case refine_1.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q r : Rat
      h : Membership.mem (setOf fun r => LT.lt ↑r ↑q) r
      ⊢ LE.le ↑r ↑q
    -/
    exact le_of_lt h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q : Rat
      w : β
      h : LT.lt w ↑q
      ⊢ Exists fun a => And (Membership.mem (LinearOrderedField.cutMap β ↑q) a) (LT. …
    -/
  · obtain ⟨q', hwq, hq⟩ := exists_rat_btwn h
    /-
      case refine_2.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q : Rat
      w : β
      h : LT.lt w ↑q
      q' : Rat
      hwq : LT.lt w ↑q'
      hq : LT.lt ↑q' ↑q
      ⊢ Exists fun a => And (Membership.mem (LinearOrderedField.cutMap β ↑q) a) (LT. …
    -/
    rw [cutMap_coe]
    /-
      case refine_2.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      q : Rat
      w : β
      h : LT.lt w ↑q
      q' : Rat
      hwq : LT.lt w ↑q'
      hq : LT.lt ↑q' ↑q
      ⊢ Exists fun a => And (Membership.mem (Set.image Rat.cast (setOf fun r => LT.l …
    -/
    exact ⟨q', ⟨_, hq, rfl⟩, hwq⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem inducedMap_zero : inducedMap α β 0 = 0 := mod_cast inducedMap_rat α β 0


@[simp]
theorem inducedMap_one : inducedMap α β 1 = 1 := mod_cast inducedMap_rat α β 1


theorem inducedMap_nonneg (ha : 0 ≤ a) : 0 ≤ inducedMap α β a :=
  (inducedMap_zero α _).ge.trans <| inducedMap_mono _ _ ha


theorem coe_lt_inducedMap_iff : (q : β) < inducedMap α β a ↔ (q : α) < a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    q : Rat
    ⊢ Iff (LT.lt (↑q) (LinearOrderedField.inducedMap α β a)) (LT.lt (↑q) a)
  -/
  refine ⟨fun h => ?_, fun hq => ?_⟩
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      q : Rat
      h : LT.lt (↑q) (LinearOrderedField.inducedMap α β a)
      ⊢ LT.lt (↑q) a
    -/
  · rw [← inducedMap_rat α] at h
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      q : Rat
      h : LT.lt (LinearOrderedField.inducedMap α β ↑q) (LinearOrderedField.inducedMa …
      ⊢ LT.lt (↑q) a
    -/
    exact (inducedMap_mono α β).reflect_lt h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      q : Rat
      hq : LT.lt (↑q) a
      ⊢ LT.lt (↑q) (LinearOrderedField.inducedMap α β a)
    -/
  · obtain ⟨q', hq, hqa⟩ := exists_rat_btwn hq
    /-
      case refine_2.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      q : Rat
      hq✝ : LT.lt (↑q) a
      q' : Rat
      hq : LT.lt ↑q ↑q'
      hqa : LT.lt (↑q') a
      ⊢ LT.lt (↑q) (LinearOrderedField.inducedMap α β a)
    -/
    apply lt_csSup_of_lt (cutMap_bddAbove β a) (coe_mem_cutMap_iff.mpr hqa)
    /-
      case refine_2.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      q : Rat
      hq✝ : LT.lt (↑q) a
      q' : Rat
      hq : LT.lt ↑q ↑q'
      hqa : LT.lt (↑q') a
      ⊢ LT.lt ↑q ↑q'
    -/
    exact mod_cast hq
    /-
      🎉 no goals
    -/


theorem lt_inducedMap_iff : b < inducedMap α β a ↔ ∃ q : ℚ, b < q ∧ (q : α) < a :=
  ⟨fun h => (exists_rat_btwn h).imp fun _ => And.imp_right coe_lt_inducedMap_iff.1,
                                         /-
                                           α : Type u_2
                                           β : Type u_3
                                           inst✝² : LinearOrderedField α
                                           inst✝¹ : ConditionallyCompleteLinearOrderedField β
                                           inst✝ : Archimedean α
                                           a : α
                                           b : β
                                           x✝ : Exists fun q => And (LT.lt b ↑q) (LT.lt (↑q) a)
                                           q : Rat
                                           hbq : LT.lt b ↑q
                                           hqa : LT.lt (↑q) a
                                           ⊢ LT.lt (↑q) (LinearOrderedField.inducedMap α β a)
                                         -/
    fun ⟨q, hbq, hqa⟩ => hbq.trans <| by rwa [coe_lt_inducedMap_iff]⟩
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem inducedMap_self (b : β) : inducedMap β β b = b :=
  eq_of_forall_rat_lt_iff_lt fun _ => coe_lt_inducedMap_iff


@[simp]
theorem inducedMap_inducedMap (a : α) : inducedMap β γ (inducedMap α β a) = inducedMap α γ a :=
  eq_of_forall_rat_lt_iff_lt fun q => by
    /-
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : LinearOrderedField α
      inst✝² : ConditionallyCompleteLinearOrderedField β
      inst✝¹ : ConditionallyCompleteLinearOrderedField γ
      inst✝ : Archimedean α
      a : α
      q : Rat
      ⊢ Iff (LT.lt (↑q) (LinearOrderedField.inducedMap β γ (LinearOrderedField.induc …
    -/
    rw [coe_lt_inducedMap_iff, coe_lt_inducedMap_iff, Iff.comm, coe_lt_inducedMap_iff]
    /-
      🎉 no goals
    -/


theorem inducedMap_inv_self (b : β) : inducedMap γ β (inducedMap β γ b) = b := by
  /-
    β : Type u_3
    γ : Type u_4
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : ConditionallyCompleteLinearOrderedField γ
    b : β
    ⊢ Eq (LinearOrderedField.inducedMap γ β (LinearOrderedField.inducedMap β γ b)) b
  -/
  rw [inducedMap_inducedMap, inducedMap_self]
  /-
    🎉 no goals
  -/


theorem inducedMap_add (x y : α) :
    inducedMap α β (x + y) = inducedMap α β x + inducedMap α β y := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    x y : α
    ⊢ Eq (LinearOrderedField.inducedMap α β (HAdd.hAdd x y)) (HAdd.hAdd (LinearOrd …
  -/
  rw [inducedMap, cutMap_add]
  exact csSup_add (cutMap_nonempty β x) (cutMap_bddAbove β x) (cutMap_nonempty β y)
    (cutMap_bddAbove β y)


/-- Preparatory lemma for `inducedOrderRingHom`. -/
theorem le_inducedMap_mul_self_of_mem_cutMap (ha : 0 < a) (b : β) (hb : b ∈ cutMap β (a * a)) :
    b ≤ inducedMap α β a * inducedMap α β a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hb : Membership.mem (LinearOrderedField.cutMap β (HMul.hMul a a)) b
    ⊢ LE.le b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedField …
  -/
  obtain ⟨q, hb, rfl⟩ := hb
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    q : Rat
    hb : Membership.mem (setOf fun t => LT.lt (↑t) (HMul.hMul a a)) q
    ⊢ LE.le (↑q) (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedFi …
  -/
  obtain ⟨q', hq', hqq', hqa⟩ := exists_rat_pow_btwn two_ne_zero hb (mul_self_pos.2 ha.ne')
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    q : Rat
    hb : Membership.mem (setOf fun t => LT.lt (↑t) (HMul.hMul a a)) q
    q' : Rat
    hq' : LT.lt 0 q'
    hqq' : LT.lt (↑q) (HPow.hPow (↑q') 2)
    hqa : LT.lt (HPow.hPow (↑q') 2) (HMul.hMul a a)
    ⊢ LE.le (↑q) (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedFi …
  -/
  trans (q' : β) ^ 2
    /-
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      ha : LT.lt 0 a
      q : Rat
      hb : Membership.mem (setOf fun t => LT.lt (↑t) (HMul.hMul a a)) q
      q' : Rat
      hq' : LT.lt 0 q'
      hqq' : LT.lt (↑q) (HPow.hPow (↑q') 2)
      hqa : LT.lt (HPow.hPow (↑q') 2) (HMul.hMul a a)
      ⊢ LE.le (↑q) (HPow.hPow (↑q') 2)
    -/
  · exact mod_cast hqq'.le
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      ha : LT.lt 0 a
      q : Rat
      hb : Membership.mem (setOf fun t => LT.lt (↑t) (HMul.hMul a a)) q
      q' : Rat
      hq' : LT.lt 0 q'
      hqq' : LT.lt (↑q) (HPow.hPow (↑q') 2)
      hqa : LT.lt (HPow.hPow (↑q') 2) (HMul.hMul a a)
      ⊢ LE.le (HPow.hPow (↑q') 2) (HMul.hMul (LinearOrderedField.inducedMap α β a) ( …
    -/
  · rw [pow_two] at hqa ⊢
    exact mul_self_le_mul_self (mod_cast hq'.le)
      (le_csSup (cutMap_bddAbove β a) <|
        coe_mem_cutMap_iff.2 <| lt_of_mul_self_lt_mul_self₀ ha.le hqa)


/-- Preparatory lemma for `inducedOrderRingHom`. -/
theorem exists_mem_cutMap_mul_self_of_lt_inducedMap_mul_self (ha : 0 < a) (b : β)
    (hba : b < inducedMap α β a * inducedMap α β a) : ∃ c ∈ cutMap β (a * a), b < c := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    ⊢ Exists fun c => And (Membership.mem (LinearOrderedField.cutMap β (HMul.hMul  …
  -/
  obtain hb | hb := lt_or_le b 0
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      ha : LT.lt 0 a
      b : β
      hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
      hb : LT.lt b 0
      ⊢ Exists fun c => And (Membership.mem (LinearOrderedField.cutMap β (HMul.hMul  …
    -/
  · refine ⟨0, ?_, hb⟩
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      ha : LT.lt 0 a
      b : β
      hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
      hb : LT.lt b 0
      ⊢ Membership.mem (LinearOrderedField.cutMap β (HMul.hMul a a)) 0
    -/
    rw [← Rat.cast_zero, coe_mem_cutMap_iff, Rat.cast_zero]
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝² : LinearOrderedField α
      inst✝¹ : ConditionallyCompleteLinearOrderedField β
      inst✝ : Archimedean α
      a : α
      ha : LT.lt 0 a
      b : β
      hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
      hb : LT.lt b 0
      ⊢ LT.lt 0 (HMul.hMul a a)
    -/
    exact mul_self_pos.2 ha.ne'
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    ⊢ Exists fun c => And (Membership.mem (LinearOrderedField.cutMap β (HMul.hMul  …
  -/
  obtain ⟨q, hq, hbq, hqa⟩ := exists_rat_pow_btwn two_ne_zero hba (hb.trans_lt hba)
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    q : Rat
    hq : LT.lt 0 q
    hbq : LT.lt b (HPow.hPow (↑q) 2)
    hqa : LT.lt (HPow.hPow (↑q) 2) (HMul.hMul (LinearOrderedField.inducedMap α β a …
    ⊢ Exists fun c => And (Membership.mem (LinearOrderedField.cutMap β (HMul.hMul  …
  -/
  rw [← cast_pow] at hbq
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    q : Rat
    hq : LT.lt 0 q
    hbq : LT.lt b ↑(HPow.hPow q 2)
    hqa : LT.lt (HPow.hPow (↑q) 2) (HMul.hMul (LinearOrderedField.inducedMap α β a …
    ⊢ Exists fun c => And (Membership.mem (LinearOrderedField.cutMap β (HMul.hMul  …
  -/
  refine ⟨(q ^ 2 : ℚ), coe_mem_cutMap_iff.2 ?_, hbq⟩
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    q : Rat
    hq : LT.lt 0 q
    hbq : LT.lt b ↑(HPow.hPow q 2)
    hqa : LT.lt (HPow.hPow (↑q) 2) (HMul.hMul (LinearOrderedField.inducedMap α β a …
    ⊢ LT.lt (↑(HPow.hPow q 2)) (HMul.hMul a a)
  -/
  rw [pow_two] at hqa ⊢
  /-
    case inr.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    q : Rat
    hq : LT.lt 0 q
    hbq : LT.lt b ↑(HPow.hPow q 2)
    hqa : LT.lt (HMul.hMul ↑q ↑q) (HMul.hMul (LinearOrderedField.inducedMap α β a) …
    ⊢ LT.lt (↑(HMul.hMul q q)) (HMul.hMul a a)
  -/
  push_cast
  obtain ⟨q', hq', hqa'⟩ := lt_inducedMap_iff.1 (lt_of_mul_self_lt_mul_self₀
    (inducedMap_nonneg ha.le) hqa)
  /-
    case inr.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : LinearOrderedField α
    inst✝¹ : ConditionallyCompleteLinearOrderedField β
    inst✝ : Archimedean α
    a : α
    ha : LT.lt 0 a
    b : β
    hba : LT.lt b (HMul.hMul (LinearOrderedField.inducedMap α β a) (LinearOrderedF …
    hb : LE.le 0 b
    q : Rat
    hq : LT.lt 0 q
    hbq : LT.lt b ↑(HPow.hPow q 2)
    hqa : LT.lt (HMul.hMul ↑q ↑q) (HMul.hMul (LinearOrderedField.inducedMap α β a) …
    q' : Rat
    hq' : LT.lt ↑q ↑q'
    hqa' : LT.lt (↑q') a
    ⊢ LT.lt (HMul.hMul ↑q ↑q) (HMul.hMul a a)
  -/
  exact mul_self_lt_mul_self (mod_cast hq.le) (hqa'.trans' <| by assumption_mod_cast)
  /-
    🎉 no goals
  -/


/-- `inducedMap` as an additive homomorphism. -/
def inducedAddHom : α →+ β :=
  ⟨⟨inducedMap α β, inducedMap_zero α β⟩, inducedMap_add α β⟩


/-- `inducedMap` as an `OrderRingHom`. -/
@[simps!]
def inducedOrderRingHom : α →+*o β :=
  { AddMonoidHom.mkRingHomOfMulSelfOfTwoNeZero (inducedAddHom α β) (by
      suffices ∀ x, 0 < x → inducedAddHom α β (x * x) = inducedAddHom α β x * inducedAddHom α β x by
        intro x
        obtain h | rfl | h := lt_trichotomy x 0
        · convert this (-x) (neg_pos.2 h) using 1
          · rw [neg_mul, mul_neg, neg_neg]
          · simp_rw [AddMonoidHom.map_neg, neg_mul, mul_neg, neg_neg]
        · simp only [mul_zero, AddMonoidHom.map_zero]
        · exact this x h
        -- prove that the (Sup of rationals less than x) ^ 2 is the Sup of the set of rationals less
        -- than (x ^ 2) by showing it is an upper bound and any smaller number is not an upper bound
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝³ : LinearOrderedField α
        inst✝² : ConditionallyCompleteLinearOrderedField β
        inst✝¹ : ConditionallyCompleteLinearOrderedField γ
        inst✝ : Archimedean α
        a : α
        b : β
        q : Rat
        ⊢ ∀ (x : α), LT.lt 0 x → Eq ((LinearOrderedField.inducedAddHom α β) (HMul.hMul …
      -/
      refine fun x hx => csSup_eq_of_forall_le_of_forall_lt_exists_gt (cutMap_nonempty β _) ?_ ?_
        /-
          case refine_1
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          x : α
          hx : LT.lt 0 x
          ⊢ ∀ (a : β), Membership.mem (LinearOrderedField.cutMap β (HMul.hMul x x)) a →  …
        -/
      · exact le_inducedMap_mul_self_of_mem_cutMap hx
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          x : α
          hx : LT.lt 0 x
          ⊢ ∀ (w : β), LT.lt w (HMul.hMul ((LinearOrderedField.inducedAddHom α β) x) ((L …
        -/
      · exact exists_mem_cutMap_mul_self_of_lt_inducedMap_mul_self hx)
        /-
          🎉 no goals
        -/
      (two_ne_zero) (inducedMap_one _ _) with
    monotone' := inducedMap_mono _ _ }


/-- The isomorphism of ordered rings between two conditionally complete linearly ordered fields. -/
def inducedOrderRingIso : β ≃+*o γ :=
  { inducedOrderRingHom β γ with
    invFun := inducedMap γ β
    left_inv := inducedMap_inv_self _ _
    right_inv := inducedMap_inv_self _ _
    map_le_map_iff' := by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝³ : LinearOrderedField α
        inst✝² : ConditionallyCompleteLinearOrderedField β
        inst✝¹ : ConditionallyCompleteLinearOrderedField γ
        inst✝ : Archimedean α
        a : α
        b : β
        q : Rat
        ⊢ ∀ {a b : β}, Iff (LE.le ({ toFun := (↑↑__src✝.toRingHom).toFun, invFun := Li …
      -/
      dsimp
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝³ : LinearOrderedField α
        inst✝² : ConditionallyCompleteLinearOrderedField β
        inst✝¹ : ConditionallyCompleteLinearOrderedField γ
        inst✝ : Archimedean α
        a : α
        b : β
        q : Rat
        ⊢ ∀ {a b : β}, Iff (LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRing …
      -/
      refine ⟨fun h => ?_, fun h => inducedMap_mono _ _ h⟩
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝³ : LinearOrderedField α
        inst✝² : ConditionallyCompleteLinearOrderedField β
        inst✝¹ : ConditionallyCompleteLinearOrderedField γ
        inst✝ : Archimedean α
        a : α
        b : β
        q : Rat
        a✝ b✝ : β
        h : LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRingHom a✝) ((Linear …
        ⊢ LE.le a✝ b✝
      -/
      convert inducedMap_mono γ β h <;>
        /-
          case h.e'_3
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          a✝ b✝ : β
          h : LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRingHom a✝) ((Linear …
          ⊢ Eq a✝ (LinearOrderedField.inducedMap γ β ((LinearOrderedField.inducedOrderRi …
        -/
        /-
          case h.e'_3
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          a✝ b✝ : β
          h : LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRingHom a✝) ((Linear …
          ⊢ Eq a✝ (LinearOrderedField.inducedMap γ β ({ toFun := LinearOrderedField.indu …
        -/
        /-
          case h.e'_3
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          a✝ b✝ : β
          h : LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRingHom a✝) ((Linear …
          ⊢ Eq a✝ (LinearOrderedField.inducedMap γ β (LinearOrderedField.inducedMap β γ  …
        -/
        /-
          🎉 no goals
        -/
        dsimp
        /-
          case h.e'_4
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝³ : LinearOrderedField α
          inst✝² : ConditionallyCompleteLinearOrderedField β
          inst✝¹ : ConditionallyCompleteLinearOrderedField γ
          inst✝ : Archimedean α
          a : α
          b : β
          q : Rat
          a✝ b✝ : β
          h : LE.le ((LinearOrderedField.inducedOrderRingHom β γ).toRingHom a✝) ((Linear …
          ⊢ Eq b✝ (LinearOrderedField.inducedMap γ β (LinearOrderedField.inducedMap β γ  …
        -/
        rw [inducedMap_inv_self β γ _] }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_inducedOrderRingIso : ⇑(inducedOrderRingIso β γ) = inducedMap β γ := rfl


@[simp]
theorem inducedOrderRingIso_symm : (inducedOrderRingIso β γ).symm = inducedOrderRingIso γ β := rfl


@[simp]
theorem inducedOrderRingIso_self : inducedOrderRingIso β β = OrderRingIso.refl β :=
  OrderRingIso.ext inducedMap_self


/-- There is a unique ordered ring homomorphism from an archimedean linear ordered field to a
conditionally complete linear ordered field. -/
instance uniqueOrderRingHom : Unique (α →+*o β) :=
  uniqueOfSubsingleton <| inducedOrderRingHom α β


/-- There is a unique ordered ring isomorphism between two conditionally complete linear ordered
fields. -/
instance uniqueOrderRingIso : Unique (β ≃+*o γ) :=
  uniqueOfSubsingleton <| inducedOrderRingIso β γ


theorem ringHom_monotone (hR : ∀ r : R, 0 ≤ r → ∃ s : R, s ^ 2 = r) (f : R →+* S) : Monotone f :=
  (monotone_iff_map_nonneg f).2 fun r h => by
    /-
      R : Type u_5
      S : Type u_6
      inst✝¹ : OrderedRing R
      inst✝ : LinearOrderedRing S
      hR : ∀ (r : R), LE.le 0 r → Exists fun s => Eq (HPow.hPow s 2) r
      f : RingHom R S
      r : R
      h : LE.le 0 r
      ⊢ LE.le 0 (f r)
    -/
    obtain ⟨s, rfl⟩ := hR r h; rw [map_pow]; apply sq_nonneg
                                             /-
                                               🎉 no goals
                                             -/


