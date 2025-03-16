instance : Add (HahnSeries Γ R) where
  add x y :=
    { coeff := x.coeff + y.coeff
      isPWO_support' := (x.isPWO_support.union y.isPWO_support).mono (Function.support_add _ _) }


instance : AddMonoid (HahnSeries Γ R) where
  zero := 0
  add := (· + ·)
  nsmul := nsmulRec
  add_assoc x y z := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x y z : HahnSeries Γ R
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd x y) z) (HAdd.hAdd x (HAdd.hAdd y z))
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x y z : HahnSeries Γ R
      x✝ : Γ
      ⊢ Eq ((HAdd.hAdd (HAdd.hAdd x y) z).coeff x✝) ((HAdd.hAdd x (HAdd.hAdd y z)).c …
    -/
    apply add_assoc
    /-
      🎉 no goals
    -/
  zero_add x := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : HahnSeries Γ R
      ⊢ Eq (HAdd.hAdd 0 x) x
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : HahnSeries Γ R
      x✝ : Γ
      ⊢ Eq ((HAdd.hAdd 0 x).coeff x✝) (x.coeff x✝)
    -/
    apply zero_add
    /-
      🎉 no goals
    -/
  add_zero x := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : HahnSeries Γ R
      ⊢ Eq (HAdd.hAdd x 0) x
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : HahnSeries Γ R
      x✝ : Γ
      ⊢ Eq ((HAdd.hAdd x 0).coeff x✝) (x.coeff x✝)
    -/
    apply add_zero
    /-
      🎉 no goals
    -/


@[simp]
theorem add_coeff' {x y : HahnSeries Γ R} : (x + y).coeff = x.coeff + y.coeff :=
  rfl


theorem add_coeff {x y : HahnSeries Γ R} {a : Γ} : (x + y).coeff a = x.coeff a + y.coeff a :=
  rfl


@[simp]
theorem nsmul_coeff {x : HahnSeries Γ R} {n : ℕ} : (n • x).coeff = n • x.coeff := by
  induction n with
  | zero => simp
  | succ n ih => simp [add_nsmul, ih]


@[simp]
protected lemma map_add [AddMonoid S] (f : R →+ S) {x y : HahnSeries Γ R} :
    ((x + y).map f : HahnSeries Γ S) = x.map f + y.map f := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : AddMonoid R
    inst✝ : AddMonoid S
    f : AddMonoidHom R S
    x y : HahnSeries Γ R
    ⊢ Eq ((HAdd.hAdd x y).map f) (HAdd.hAdd (x.map f) (y.map f))
  -/
  ext; simp
       /-
         🎉 no goals
       -/

/--
`addOppositeEquiv` is an additive monoid isomorphism between
Hahn series over `Γ` with coefficients in the opposite additive monoid `Rᵃᵒᵖ`
and the additive opposite of Hahn series over `Γ` with coefficients `R`.
-/
@[simps (config := .lemmasOnly)]
def addOppositeEquiv : HahnSeries Γ (Rᵃᵒᵖ) ≃+ (HahnSeries Γ R)ᵃᵒᵖ where
                                               /-
                                                 Γ : Type u_1
                                                 Γ' : Type u_2
                                                 R : Type u_3
                                                 S : Type u_4
                                                 U : Type u_5
                                                 V : Type u_6
                                                 inst✝¹ : PartialOrder Γ
                                                 inst✝ : AddMonoid R
                                                 x : HahnSeries Γ (AddOpposite R)
                                                 ⊢ (Function.support fun a => AddOpposite.unop (x.coeff a)).IsPWO
                                               -/
  toFun x := .op ⟨fun a ↦ (x.coeff a).unop, by convert x.isPWO_support; ext; simp⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                /-
                                                  Γ : Type u_1
                                                  Γ' : Type u_2
                                                  R : Type u_3
                                                  S : Type u_4
                                                  U : Type u_5
                                                  V : Type u_6
                                                  inst✝¹ : PartialOrder Γ
                                                  inst✝ : AddMonoid R
                                                  x : AddOpposite (HahnSeries Γ R)
                                                  ⊢ (Function.support fun a => AddOpposite.op ((AddOpposite.unop x).coeff a)).Is …
                                                -/
  invFun x := ⟨fun a ↦ .op (x.unop.coeff a), by convert x.unop.isPWO_support; ext; simp⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                   /-
                     Γ : Type u_1
                     Γ' : Type u_2
                     R : Type u_3
                     S : Type u_4
                     U : Type u_5
                     V : Type u_6
                     inst✝¹ : PartialOrder Γ
                     inst✝ : AddMonoid R
                     x : HahnSeries Γ (AddOpposite R)
                     ⊢ Eq ((fun x => { coeff := fun a => AddOpposite.op ((AddOpposite.unop x).coeff …
                   -/
  left_inv x := by ext; simp
                        /-
                          🎉 no goals
                        -/
  right_inv x := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : AddOpposite (HahnSeries Γ R)
      ⊢ Eq ((fun x => AddOpposite.op { coeff := fun a => AddOpposite.unop (x.coeff a …
    -/
    apply AddOpposite.unop_injective
    /-
      case a
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : AddOpposite (HahnSeries Γ R)
      ⊢ Eq (AddOpposite.unop ((fun x => AddOpposite.op { coeff := fun a => AddOpposi …
    -/
    ext
    /-
      case a.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x : AddOpposite (HahnSeries Γ R)
      x✝ : Γ
      ⊢ Eq ((AddOpposite.unop ((fun x => AddOpposite.op { coeff := fun a => AddOppos …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' x y := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x y : HahnSeries Γ (AddOpposite R)
      ⊢ Eq ({ toFun := fun x => AddOpposite.op { coeff := fun a => AddOpposite.unop  …
    -/
    apply AddOpposite.unop_injective
    /-
      case a
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x y : HahnSeries Γ (AddOpposite R)
      ⊢ Eq (AddOpposite.unop ({ toFun := fun x => AddOpposite.op { coeff := fun a => …
    -/
    ext
    /-
      case a.coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V : Type u_6
      inst✝¹ : PartialOrder Γ
      inst✝ : AddMonoid R
      x y : HahnSeries Γ (AddOpposite R)
      x✝ : Γ
      ⊢ Eq ((AddOpposite.unop ({ toFun := fun x => AddOpposite.op { coeff := fun a = …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
lemma addOppositeEquiv_support (x : HahnSeries Γ (Rᵃᵒᵖ)) :
    (addOppositeEquiv x).unop.support = x.support := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : HahnSeries Γ (AddOpposite R)
    ⊢ Eq (AddOpposite.unop (HahnSeries.addOppositeEquiv x)).support x.support
  -/
  ext
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : HahnSeries Γ (AddOpposite R)
    x✝ : Γ
    ⊢ Iff (Membership.mem (AddOpposite.unop (HahnSeries.addOppositeEquiv x)).suppo …
  -/
  simp [addOppositeEquiv_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma addOppositeEquiv_symm_support (x : (HahnSeries Γ R)ᵃᵒᵖ) :
    (addOppositeEquiv.symm x).support = x.unop.support := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : AddOpposite (HahnSeries Γ R)
    ⊢ Eq (HahnSeries.addOppositeEquiv.symm x).support (AddOpposite.unop x).support
  -/
  rw [← addOppositeEquiv_support, AddEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma addOppositeEquiv_orderTop (x : HahnSeries Γ (Rᵃᵒᵖ)) :
    (addOppositeEquiv x).unop.orderTop = x.orderTop := by
  classical
  simp only [orderTop, AddOpposite.unop_op, mk_eq_zero, EmbeddingLike.map_eq_zero_iff,
    addOppositeEquiv_support, ne_eq]
  simp only [addOppositeEquiv_apply, AddOpposite.unop_op, mk_eq_zero, zero_coeff]
  simp_rw [HahnSeries.ext_iff, funext_iff]
  simp only [Pi.zero_apply, AddOpposite.unop_eq_zero_iff, zero_coeff]


@[simp]
lemma addOppositeEquiv_symm_orderTop (x : (HahnSeries Γ R)ᵃᵒᵖ) :
    (addOppositeEquiv.symm x).orderTop = x.unop.orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : AddOpposite (HahnSeries Γ R)
    ⊢ Eq (HahnSeries.addOppositeEquiv.symm x).orderTop (AddOpposite.unop x).orderTop
  -/
  rw [← addOppositeEquiv_orderTop, AddEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma addOppositeEquiv_leadingCoeff (x : HahnSeries Γ (Rᵃᵒᵖ)) :
    (addOppositeEquiv x).unop.leadingCoeff = x.leadingCoeff.unop := by
  classical
  simp only [leadingCoeff, AddOpposite.unop_op, mk_eq_zero, EmbeddingLike.map_eq_zero_iff,
    addOppositeEquiv_support, ne_eq]
  simp only [addOppositeEquiv_apply, AddOpposite.unop_op, mk_eq_zero, zero_coeff]
  simp_rw [HahnSeries.ext_iff, funext_iff]
  simp only [Pi.zero_apply, AddOpposite.unop_eq_zero_iff, zero_coeff]
  split <;> rfl


@[simp]
lemma addOppositeEquiv_symm_leadingCoeff (x : (HahnSeries Γ R)ᵃᵒᵖ) :
    (addOppositeEquiv.symm x).leadingCoeff = .op x.unop.leadingCoeff := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : AddOpposite (HahnSeries Γ R)
    ⊢ Eq (HahnSeries.addOppositeEquiv.symm x).leadingCoeff (AddOpposite.op (AddOpp …
  -/
  apply AddOpposite.unop_injective
  /-
    case a
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x : AddOpposite (HahnSeries Γ R)
    ⊢ Eq (AddOpposite.unop (HahnSeries.addOppositeEquiv.symm x).leadingCoeff) (Add …
  -/
  rw [← addOppositeEquiv_leadingCoeff, AddEquiv.apply_symm_apply, AddOpposite.unop_op]
  /-
    🎉 no goals
  -/


theorem support_add_subset {x y : HahnSeries Γ R} : support (x + y) ⊆ support x ∪ support y :=
  fun a ha => by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x y : HahnSeries Γ R
    a : Γ
    ha : Membership.mem (HAdd.hAdd x y).support a
    ⊢ Membership.mem (Union.union x.support y.support) a
  -/
  rw [mem_support, add_coeff] at ha
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x y : HahnSeries Γ R
    a : Γ
    ha : Ne (HAdd.hAdd (x.coeff a) (y.coeff a)) 0
    ⊢ Membership.mem (Union.union x.support y.support) a
  -/
  rw [Set.mem_union, mem_support, mem_support]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x y : HahnSeries Γ R
    a : Γ
    ha : Ne (HAdd.hAdd (x.coeff a) (y.coeff a)) 0
    ⊢ Or (Ne (x.coeff a) 0) (Ne (y.coeff a) 0)
  -/
  contrapose! ha
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddMonoid R
    x y : HahnSeries Γ R
    a : Γ
    ha : And (Eq (x.coeff a) 0) (Eq (y.coeff a) 0)
    ⊢ Eq (HAdd.hAdd (x.coeff a) (y.coeff a)) 0
  -/
  rw [ha.1, ha.2, add_zero]
  /-
    🎉 no goals
  -/


protected theorem min_le_min_add {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R} (hx : x ≠ 0)
    (hy : y ≠ 0) (hxy : x + y ≠ 0) :
    min (Set.IsWF.min x.isWF_support (support_nonempty_iff.2 hx))
      (Set.IsWF.min y.isWF_support (support_nonempty_iff.2 hy)) ≤
      Set.IsWF.min (x + y).isWF_support (support_nonempty_iff.2 hxy) := by
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hx : Ne x 0
    hy : Ne y 0
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (Min.min (⋯.min ⋯) (⋯.min ⋯)) (⋯.min ⋯)
  -/
  rw [← Set.IsWF.min_union]
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hx : Ne x 0
    hy : Ne y 0
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (⋯.min ⋯) (⋯.min ⋯)
  -/
  exact Set.IsWF.min_le_min_of_subset (support_add_subset (x := x) (y := y))
  /-
    🎉 no goals
  -/


theorem min_orderTop_le_orderTop_add {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R} :
    min x.orderTop y.orderTop ≤ (x + y).orderTop := by
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    ⊢ LE.le (Min.min x.orderTop y.orderTop) (HAdd.hAdd x y).orderTop
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    ⊢ LE.le (Min.min x.orderTop y.orderTop) (HAdd.hAdd x y).orderTop
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ LE.le (Min.min x.orderTop y.orderTop) (HAdd.hAdd x y).orderTop
  -/
  by_cases hxy : x + y = 0; · simp [hxy]
                              /-
                                🎉 no goals
                              -/
  rw [orderTop_of_ne hx, orderTop_of_ne hy, orderTop_of_ne hxy, ← WithTop.coe_min,
    WithTop.coe_le_coe]
  /-
    case neg
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hxy : Not (Eq (HAdd.hAdd x y) 0)
    ⊢ LE.le (Min.min (⋯.min ⋯) (⋯.min ⋯)) (⋯.min ⋯)
  -/
  exact HahnSeries.min_le_min_add hx hy hxy
  /-
    🎉 no goals
  -/


theorem min_order_le_order_add {Γ} [Zero Γ] [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : x + y ≠ 0) : min x.order y.order ≤ (x + y).order := by
  /-
    R : Type u_3
    inst✝² : AddMonoid R
    Γ : Type u_7
    inst✝¹ : Zero Γ
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (Min.min x.order y.order) (HAdd.hAdd x y).order
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    inst✝² : AddMonoid R
    Γ : Type u_7
    inst✝¹ : Zero Γ
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    ⊢ LE.le (Min.min x.order y.order) (HAdd.hAdd x y).order
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    inst✝² : AddMonoid R
    Γ : Type u_7
    inst✝¹ : Zero Γ
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ LE.le (Min.min x.order y.order) (HAdd.hAdd x y).order
  -/
  rw [order_of_ne hx, order_of_ne hy, order_of_ne hxy]
  /-
    case neg
    R : Type u_3
    inst✝² : AddMonoid R
    Γ : Type u_7
    inst✝¹ : Zero Γ
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : Ne (HAdd.hAdd x y) 0
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ LE.le (Min.min (⋯.min ⋯) (⋯.min ⋯)) (⋯.min ⋯)
  -/
  exact HahnSeries.min_le_min_add hx hy hxy
  /-
    🎉 no goals
  -/


theorem orderTop_add_eq_left {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : x.orderTop < y.orderTop) : (x + y).orderTop = x.orderTop := by
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HAdd.hAdd x y).orderTop x.orderTop
  -/
  have hx : x ≠ 0 := ne_zero_iff_orderTop.mpr hxy.ne_top
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    hx : Ne x 0
    ⊢ Eq (HAdd.hAdd x y).orderTop x.orderTop
  -/
  let g : Γ := Set.IsWF.min x.isWF_support (support_nonempty_iff.2 hx)
  have hcxyne : (x + y).coeff g ≠ 0 := by
    rw [add_coeff, coeff_eq_zero_of_lt_orderTop (lt_of_eq_of_lt (orderTop_of_ne hx).symm hxy),
      add_zero]
    exact coeff_orderTop_ne (orderTop_of_ne hx)
  have hxyx : (x + y).orderTop ≤ x.orderTop := by
    rw [orderTop_of_ne hx]
    exact orderTop_le_of_coeff_ne_zero hcxyne
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    hx : Ne x 0
    g : Γ := ⋯.min ⋯
    hcxyne : Ne ((HAdd.hAdd x y).coeff g) 0
    hxyx : LE.le (HAdd.hAdd x y).orderTop x.orderTop
    ⊢ Eq (HAdd.hAdd x y).orderTop x.orderTop
  -/
  exact le_antisymm hxyx (le_of_eq_of_le (min_eq_left_of_lt hxy).symm min_orderTop_le_orderTop_add)
  /-
    🎉 no goals
  -/


theorem orderTop_add_eq_right {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : y.orderTop < x.orderTop) : (x + y).orderTop = y.orderTop := by
  simpa [← map_add, ← AddOpposite.op_add, hxy] using orderTop_add_eq_left
    (x := addOppositeEquiv.symm (.op y))
    (y := addOppositeEquiv.symm (.op x))


theorem leadingCoeff_add_eq_left {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : x.orderTop < y.orderTop) : (x + y).leadingCoeff = x.leadingCoeff := by
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
  -/
  have hx : x ≠ 0 := ne_zero_iff_orderTop.mpr hxy.ne_top
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    hx : Ne x 0
    ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
  -/
  have ho : (x + y).orderTop = x.orderTop := orderTop_add_eq_left hxy
  /-
    R : Type u_3
    inst✝¹ : AddMonoid R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    hx : Ne x 0
    ho : Eq (HAdd.hAdd x y).orderTop x.orderTop
    ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
  -/
  by_cases h : x + y = 0
    /-
      case pos
      R : Type u_3
      inst✝¹ : AddMonoid R
      Γ : Type u_7
      inst✝ : LinearOrder Γ
      x y : HahnSeries Γ R
      hxy : LT.lt x.orderTop y.orderTop
      hx : Ne x 0
      ho : Eq (HAdd.hAdd x y).orderTop x.orderTop
      h : Eq (HAdd.hAdd x y) 0
      ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
    -/
  · rw [h, orderTop_zero] at ho
    /-
      case pos
      R : Type u_3
      inst✝¹ : AddMonoid R
      Γ : Type u_7
      inst✝ : LinearOrder Γ
      x y : HahnSeries Γ R
      hxy : LT.lt x.orderTop y.orderTop
      hx : Ne x 0
      ho : Eq Top.top x.orderTop
      h : Eq (HAdd.hAdd x y) 0
      ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
    -/
    rw [h, orderTop_eq_top_iff.mp ho.symm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_3
      inst✝¹ : AddMonoid R
      Γ : Type u_7
      inst✝ : LinearOrder Γ
      x y : HahnSeries Γ R
      hxy : LT.lt x.orderTop y.orderTop
      hx : Ne x 0
      ho : Eq (HAdd.hAdd x y).orderTop x.orderTop
      h : Not (Eq (HAdd.hAdd x y) 0)
      ⊢ Eq (HAdd.hAdd x y).leadingCoeff x.leadingCoeff
    -/
  · rw [orderTop_of_ne h, orderTop_of_ne hx, WithTop.coe_eq_coe] at ho
    rw [leadingCoeff_of_ne h, leadingCoeff_of_ne hx, ho, add_coeff,
      coeff_eq_zero_of_lt_orderTop (lt_of_eq_of_lt (orderTop_of_ne hx).symm hxy), add_zero]


theorem leadingCoeff_add_eq_right {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : y.orderTop < x.orderTop) : (x + y).leadingCoeff = y.leadingCoeff := by
  simpa [← map_add, ← AddOpposite.op_add, hxy] using leadingCoeff_add_eq_left
    (x := addOppositeEquiv.symm (.op y))
    (y := addOppositeEquiv.symm (.op x))


/-- `single` as an additive monoid/group homomorphism -/
@[simps!]
def single.addMonoidHom (a : Γ) : R →+ HahnSeries Γ R :=
  { single a with
    map_add' := fun x y => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddMonoid R
        a : Γ
        x y : R
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (__src✝.toFun x) (__src✝.toFun  …
      -/
      ext b
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddMonoid R
        a : Γ
        x y : R
        b : Γ
        ⊢ Eq ((__src✝.toFun (HAdd.hAdd x y)).coeff b) ((HAdd.hAdd (__src✝.toFun x) (__ …
      -/
                             /-
                               🎉 no goals
                             -/
      by_cases h : b = a <;> simp [h] }
                             /-
                               🎉 no goals
                             -/


/-- `coeff g` as an additive monoid/group homomorphism -/
@[simps]
def coeff.addMonoidHom (g : Γ) : HahnSeries Γ R →+ R where
  toFun f := f.coeff g
  map_zero' := zero_coeff
  map_add' _ _ := add_coeff


theorem embDomain_add (f : Γ ↪o Γ') (x y : HahnSeries Γ R) :
    embDomain f (x + y) = embDomain f x + embDomain f y := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : AddMonoid R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    ⊢ Eq (HahnSeries.embDomain f (HAdd.hAdd x y)) (HAdd.hAdd (HahnSeries.embDomain …
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : AddMonoid R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    x y : HahnSeries Γ R
    g : Γ'
    ⊢ Eq ((HahnSeries.embDomain f (HAdd.hAdd x y)).coeff g) ((HAdd.hAdd (HahnSerie …
  -/
  by_cases hg : g ∈ Set.range f
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : AddMonoid R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x y : HahnSeries Γ R
      g : Γ'
      hg : Membership.mem (Set.range ⇑f) g
      ⊢ Eq ((HahnSeries.embDomain f (HAdd.hAdd x y)).coeff g) ((HAdd.hAdd (HahnSerie …
    -/
  · obtain ⟨a, rfl⟩ := hg
    /-
      case pos.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : AddMonoid R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x y : HahnSeries Γ R
      a : Γ
      ⊢ Eq ((HahnSeries.embDomain f (HAdd.hAdd x y)).coeff (f a)) ((HAdd.hAdd (HahnS …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : AddMonoid R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      x y : HahnSeries Γ R
      g : Γ'
      hg : Not (Membership.mem (Set.range ⇑f) g)
      ⊢ Eq ((HahnSeries.embDomain f (HAdd.hAdd x y)).coeff g) ((HAdd.hAdd (HahnSerie …
    -/
  · simp [embDomain_notin_range hg]
    /-
      🎉 no goals
    -/


instance [AddCommMonoid R] : AddCommMonoid (HahnSeries Γ R) :=
  { inferInstanceAs (AddMonoid (HahnSeries Γ R)) with
    add_comm := fun x y => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        x y : HahnSeries Γ R
        ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd y x)
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddCommMonoid R
        x y : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HAdd.hAdd x y).coeff x✝) ((HAdd.hAdd y x).coeff x✝)
      -/
      apply add_comm }
      /-
        🎉 no goals
      -/


instance : Neg (HahnSeries Γ R) where
  neg x :=
    { coeff := fun a => -x.coeff a
      isPWO_support' := by
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          S : Type u_4
          U : Type u_5
          V : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddGroup R
          x : HahnSeries Γ R
          ⊢ (Function.support fun a => Neg.neg (x.coeff a)).IsPWO
        -/
        rw [Function.support_neg]
        /-
          Γ : Type u_1
          Γ' : Type u_2
          R : Type u_3
          S : Type u_4
          U : Type u_5
          V : Type u_6
          inst✝¹ : PartialOrder Γ
          inst✝ : AddGroup R
          x : HahnSeries Γ R
          ⊢ (Function.support x.coeff).IsPWO
        -/
        exact x.isPWO_support }
        /-
          🎉 no goals
        -/


instance : AddGroup (HahnSeries Γ R) :=
  { inferInstanceAs (AddMonoid (HahnSeries Γ R)) with
    zsmul := zsmulRec
    neg_add_cancel := fun x => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddGroup R
        x : HahnSeries Γ R
        ⊢ Eq (HAdd.hAdd (Neg.neg x) x) 0
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝¹ : PartialOrder Γ
        inst✝ : AddGroup R
        x : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HAdd.hAdd (Neg.neg x) x).coeff x✝) (HahnSeries.coeff 0 x✝)
      -/
      apply neg_add_cancel }
      /-
        🎉 no goals
      -/


@[simp]
theorem neg_coeff' {x : HahnSeries Γ R} : (-x).coeff = -x.coeff :=
  rfl


theorem neg_coeff {x : HahnSeries Γ R} {a : Γ} : (-x).coeff a = -x.coeff a :=
  rfl


@[simp]
theorem support_neg {x : HahnSeries Γ R} : (-x).support = x.support := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x : HahnSeries Γ R
    ⊢ Eq (Neg.neg x).support x.support
  -/
  ext
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x : HahnSeries Γ R
    x✝ : Γ
    ⊢ Iff (Membership.mem (Neg.neg x).support x✝) (Membership.mem x.support x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma map_neg [AddGroup S] (f : R →+ S) {x : HahnSeries Γ R} :
    ((-x).map f : HahnSeries Γ S) = -(x.map f) := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : AddGroup R
    inst✝ : AddGroup S
    f : AddMonoidHom R S
    x : HahnSeries Γ R
    ⊢ Eq ((Neg.neg x).map f) (Neg.neg (x.map f))
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem orderTop_neg {x : HahnSeries Γ R} : (-x).orderTop = x.orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x : HahnSeries Γ R
    ⊢ Eq (Neg.neg x).orderTop x.orderTop
  -/
  classical simp only [orderTop, support_neg, neg_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem order_neg [Zero Γ] {f : HahnSeries Γ R} : (-f).order = f.order := by
  classical
  by_cases hf : f = 0
  · simp only [hf, neg_zero]
  simp only [order, support_neg, neg_eq_zero]


@[simp]
theorem sub_coeff' {x y : HahnSeries Γ R} : (x - y).coeff = x.coeff - y.coeff := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x y : HahnSeries Γ R
    ⊢ Eq (HSub.hSub x y).coeff (HSub.hSub x.coeff y.coeff)
  -/
  ext
  /-
    case h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x y : HahnSeries Γ R
    x✝ : Γ
    ⊢ Eq ((HSub.hSub x y).coeff x✝) (HSub.hSub x.coeff y.coeff x✝)
  -/
  simp [sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem sub_coeff {x y : HahnSeries Γ R} {a : Γ} : (x - y).coeff a = x.coeff a - y.coeff a := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : PartialOrder Γ
    inst✝ : AddGroup R
    x y : HahnSeries Γ R
    a : Γ
    ⊢ Eq ((HSub.hSub x y).coeff a) (HSub.hSub (x.coeff a) (y.coeff a))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma map_sub [AddGroup S] (f : R →+ S) {x y : HahnSeries Γ R} :
    ((x - y).map f : HahnSeries Γ S) = x.map f - y.map f := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : PartialOrder Γ
    inst✝¹ : AddGroup R
    inst✝ : AddGroup S
    f : AddMonoidHom R S
    x y : HahnSeries Γ R
    ⊢ Eq ((HSub.hSub x y).map f) (HSub.hSub (x.map f) (y.map f))
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem min_orderTop_le_orderTop_sub {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R} :
    min x.orderTop y.orderTop ≤ (x - y).orderTop := by
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    ⊢ LE.le (Min.min x.orderTop y.orderTop) (HSub.hSub x y).orderTop
  -/
  rw [sub_eq_add_neg, ← orderTop_neg (x := y)]
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    ⊢ LE.le (Min.min x.orderTop (Neg.neg y).orderTop) (HAdd.hAdd x (Neg.neg y)).or …
  -/
  exact min_orderTop_le_orderTop_add
  /-
    🎉 no goals
  -/


theorem orderTop_sub {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : x.orderTop < y.orderTop) : (x - y).orderTop = x.orderTop := by
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HSub.hSub x y).orderTop x.orderTop
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HAdd.hAdd x (Neg.neg y)).orderTop x.orderTop
  -/
  rw [← orderTop_neg (x := y)] at hxy
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop (Neg.neg y).orderTop
    ⊢ Eq (HAdd.hAdd x (Neg.neg y)).orderTop x.orderTop
  -/
  exact orderTop_add_eq_left hxy
  /-
    🎉 no goals
  -/


theorem leadingCoeff_sub {Γ} [LinearOrder Γ] {x y : HahnSeries Γ R}
    (hxy : x.orderTop < y.orderTop) : (x - y).leadingCoeff = x.leadingCoeff := by
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HSub.hSub x y).leadingCoeff x.leadingCoeff
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop y.orderTop
    ⊢ Eq (HAdd.hAdd x (Neg.neg y)).leadingCoeff x.leadingCoeff
  -/
  rw [← orderTop_neg (x := y)] at hxy
  /-
    R : Type u_3
    inst✝¹ : AddGroup R
    Γ : Type u_7
    inst✝ : LinearOrder Γ
    x y : HahnSeries Γ R
    hxy : LT.lt x.orderTop (Neg.neg y).orderTop
    ⊢ Eq (HAdd.hAdd x (Neg.neg y)).leadingCoeff x.leadingCoeff
  -/
  exact leadingCoeff_add_eq_left hxy
  /-
    🎉 no goals
  -/


instance [AddCommGroup R] : AddCommGroup (HahnSeries Γ R) :=
  { inferInstanceAs (AddCommMonoid (HahnSeries Γ R)),
    inferInstanceAs (AddGroup (HahnSeries Γ R)) with }


instance : SMul R (HahnSeries Γ V) :=
  ⟨fun r x =>
    { coeff := r • x.coeff
      isPWO_support' := x.isPWO_support.mono (Function.support_const_smul_subset r x.coeff) }⟩


@[simp]
theorem smul_coeff {r : R} {x : HahnSeries Γ V} {a : Γ} : (r • x).coeff a = r • x.coeff a :=
  rfl


instance : SMulZeroClass R (HahnSeries Γ V) :=
  { inferInstanceAs (SMul R (HahnSeries Γ V)) with
    smul_zero := by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V✝ : Type u_6
        inst✝² : PartialOrder Γ
        V : Type u_7
        inst✝¹ : Zero V
        inst✝ : SMulZeroClass R V
        ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
      -/
      intro
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V✝ : Type u_6
        inst✝² : PartialOrder Γ
        V : Type u_7
        inst✝¹ : Zero V
        inst✝ : SMulZeroClass R V
        a✝ : R
        ⊢ Eq (HSMul.hSMul a✝ 0) 0
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V✝ : Type u_6
        inst✝² : PartialOrder Γ
        V : Type u_7
        inst✝¹ : Zero V
        inst✝ : SMulZeroClass R V
        a✝ : R
        x✝ : Γ
        ⊢ Eq ((HSMul.hSMul a✝ 0).coeff x✝) (HahnSeries.coeff 0 x✝)
      -/
      simp only [smul_coeff, zero_coeff, smul_zero]}
      /-
        🎉 no goals
      -/


theorem orderTop_smul_not_lt (r : R) (x : HahnSeries Γ V) : ¬ (r • x).orderTop < x.orderTop := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : PartialOrder Γ
    V : Type u_7
    inst✝¹ : Zero V
    inst✝ : SMulZeroClass R V
    r : R
    x : HahnSeries Γ V
    ⊢ Not (LT.lt (HSMul.hSMul r x).orderTop x.orderTop)
  -/
  by_cases hrx : r • x = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      V : Type u_7
      inst✝¹ : Zero V
      inst✝ : SMulZeroClass R V
      r : R
      x : HahnSeries Γ V
      hrx : Eq (HSMul.hSMul r x) 0
      ⊢ Not (LT.lt (HSMul.hSMul r x).orderTop x.orderTop)
    -/
  · rw [hrx, orderTop_zero]
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      V : Type u_7
      inst✝¹ : Zero V
      inst✝ : SMulZeroClass R V
      r : R
      x : HahnSeries Γ V
      hrx : Eq (HSMul.hSMul r x) 0
      ⊢ Not (LT.lt Top.top x.orderTop)
    -/
    exact not_top_lt
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : PartialOrder Γ
      V : Type u_7
      inst✝¹ : Zero V
      inst✝ : SMulZeroClass R V
      r : R
      x : HahnSeries Γ V
      hrx : Not (Eq (HSMul.hSMul r x) 0)
      ⊢ Not (LT.lt (HSMul.hSMul r x).orderTop x.orderTop)
    -/
  · simp only [orderTop_of_ne hrx, orderTop_of_ne <| right_ne_zero_of_smul hrx, WithTop.coe_lt_coe]
    exact Set.IsWF.min_of_subset_not_lt_min
      (Function.support_smul_subset_right (fun _ => r) x.coeff)


theorem order_smul_not_lt [Zero Γ] (r : R) (x : HahnSeries Γ V) (h : r • x ≠ 0) :
    ¬ (r • x).order < x.order := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : PartialOrder Γ
    V : Type u_7
    inst✝² : Zero V
    inst✝¹ : SMulZeroClass R V
    inst✝ : Zero Γ
    r : R
    x : HahnSeries Γ V
    h : Ne (HSMul.hSMul r x) 0
    ⊢ Not (LT.lt (HSMul.hSMul r x).order x.order)
  -/
  have hx : x ≠ 0 := right_ne_zero_of_smul h
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : PartialOrder Γ
    V : Type u_7
    inst✝² : Zero V
    inst✝¹ : SMulZeroClass R V
    inst✝ : Zero Γ
    r : R
    x : HahnSeries Γ V
    h : Ne (HSMul.hSMul r x) 0
    hx : Ne x 0
    ⊢ Not (LT.lt (HSMul.hSMul r x).order x.order)
  -/
  simp_all only [order, dite_false]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : PartialOrder Γ
    V : Type u_7
    inst✝² : Zero V
    inst✝¹ : SMulZeroClass R V
    inst✝ : Zero Γ
    r : R
    x : HahnSeries Γ V
    h : Ne (HSMul.hSMul r x) 0
    hx : Ne x 0
    ⊢ Not (LT.lt (⋯.min ⋯) (⋯.min ⋯))
  -/
  exact Set.IsWF.min_of_subset_not_lt_min (Function.support_smul_subset_right (fun _ => r) x.coeff)
  /-
    🎉 no goals
  -/


theorem le_order_smul {Γ} [Zero Γ] [LinearOrder Γ] (r : R) (x : HahnSeries Γ V) (h : r • x ≠ 0) :
    x.order ≤ (r • x).order :=
  le_of_not_lt (order_smul_not_lt r x h)


instance : DistribMulAction R (HahnSeries Γ V) where
  smul := (· • ·)
  one_smul _ := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝ : HahnSeries Γ V
      ⊢ Eq (HSMul.hSMul 1 x✝) x✝
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝¹ : HahnSeries Γ V
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul 1 x✝¹).coeff x✝) (x✝¹.coeff x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  smul_zero _ := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝ : R
      ⊢ Eq (HSMul.hSMul x✝ 0) 0
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝¹ : R
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul x✝¹ 0).coeff x✝) (HahnSeries.coeff 0 x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  smul_add _ _ _ := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝² x✝¹ : R
      x✝ : HahnSeries Γ V
      ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
    -/
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝² : R
      x✝¹ x✝ : HahnSeries Γ V
      ⊢ Eq (HSMul.hSMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HSMul.hSMul x✝² x✝¹) (HS …
    -/
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝³ x✝² : R
      x✝¹ : HahnSeries Γ V
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul (HMul.hMul x✝³ x✝²) x✝¹).coeff x✝) ((HSMul.hSMul x✝³ (HSMul …
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝³ : PartialOrder Γ
      V : Type u_7
      inst✝² : Monoid R
      inst✝¹ : AddMonoid V
      inst✝ : DistribMulAction R V
      x✝³ : R
      x✝² x✝¹ : HahnSeries Γ V
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul x✝³ (HAdd.hAdd x✝² x✝¹)).coeff x✝) ((HAdd.hAdd (HSMul.hSMul …
    -/
    simp [smul_add]
    /-
      🎉 no goals
    -/
  mul_smul _ _ _ := by
    ext
    simp [mul_smul]


instance [SMul R S] [IsScalarTower R S V] : IsScalarTower R S (HahnSeries Γ V) :=
  ⟨fun r s a => by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S✝ : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝⁷ : PartialOrder Γ
      V : Type u_7
      inst✝⁶ : Monoid R
      inst✝⁵ : AddMonoid V
      inst✝⁴ : DistribMulAction R V
      S : Type u_8
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S V
      inst✝¹ : SMul R S
      inst✝ : IsScalarTower R S V
      r : R
      s : S
      a : HahnSeries Γ V
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) a) (HSMul.hSMul r (HSMul.hSMul s a))
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S✝ : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝⁷ : PartialOrder Γ
      V : Type u_7
      inst✝⁶ : Monoid R
      inst✝⁵ : AddMonoid V
      inst✝⁴ : DistribMulAction R V
      S : Type u_8
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S V
      inst✝¹ : SMul R S
      inst✝ : IsScalarTower R S V
      r : R
      s : S
      a : HahnSeries Γ V
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul (HSMul.hSMul r s) a).coeff x✝) ((HSMul.hSMul r (HSMul.hSMul …
    -/
    simp⟩
    /-
      🎉 no goals
    -/


instance [SMulCommClass R S V] : SMulCommClass R S (HahnSeries Γ V) :=
  ⟨fun r s a => by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S✝ : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝⁶ : PartialOrder Γ
      V : Type u_7
      inst✝⁵ : Monoid R
      inst✝⁴ : AddMonoid V
      inst✝³ : DistribMulAction R V
      S : Type u_8
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S V
      inst✝ : SMulCommClass R S V
      r : R
      s : S
      a : HahnSeries Γ V
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul s a)) (HSMul.hSMul s (HSMul.hSMul r a))
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S✝ : Type u_4
      U : Type u_5
      V✝ : Type u_6
      inst✝⁶ : PartialOrder Γ
      V : Type u_7
      inst✝⁵ : Monoid R
      inst✝⁴ : AddMonoid V
      inst✝³ : DistribMulAction R V
      S : Type u_8
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S V
      inst✝ : SMulCommClass R S V
      r : R
      s : S
      a : HahnSeries Γ V
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul r (HSMul.hSMul s a)).coeff x✝) ((HSMul.hSMul s (HSMul.hSMul …
    -/
    simp [smul_comm]⟩
    /-
      🎉 no goals
    -/


instance : Module R (HahnSeries Γ V) :=
  { inferInstanceAs (DistribMulAction R (HahnSeries Γ V)) with
    zero_smul := fun _ => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        x✝ : HahnSeries Γ V
        ⊢ Eq (HSMul.hSMul 0 x✝) 0
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        x✝¹ : HahnSeries Γ V
        x✝ : Γ
        ⊢ Eq ((HSMul.hSMul 0 x✝¹).coeff x✝) (HahnSeries.coeff 0 x✝)
      -/
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        x✝² x✝¹ : R
        x✝ : HahnSeries Γ V
        ⊢ Eq (HSMul.hSMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HSMul.hSMul x✝² x✝) (HSM …
      -/
      simp
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        x✝³ x✝² : R
        x✝¹ : HahnSeries Γ V
        x✝ : Γ
        ⊢ Eq ((HSMul.hSMul (HAdd.hAdd x✝³ x✝²) x✝¹).coeff x✝) ((HAdd.hAdd (HSMul.hSMul …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    add_smul := fun _ _ _ => by
      ext
      simp [add_smul] }


/-- `single` as a linear map -/
@[simps]
def single.linearMap (a : Γ) : V →ₗ[R] HahnSeries Γ V :=
  { single.addMonoidHom a with
    map_smul' := fun r s => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        a : Γ
        r : R
        s : V
        ⊢ Eq ({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r s)) (HS …
      -/
      ext b
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        U : Type u_5
        V : Type u_6
        inst✝³ : PartialOrder Γ
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid V
        inst✝ : Module R V
        a : Γ
        r : R
        s : V
        b : Γ
        ⊢ Eq (({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r s)).co …
      -/
                             /-
                               🎉 no goals
                             -/
      by_cases h : b = a <;> simp [h] }
                             /-
                               🎉 no goals
                             -/


/-- `coeff g` as a linear map -/
@[simps]
def coeff.linearMap (g : Γ) : HahnSeries Γ V →ₗ[R] V :=
  { coeff.addMonoidHom g with map_smul' := fun _ _ => rfl }


@[simp]
protected lemma map_smul [AddCommMonoid U] [Module R U] (f : U →ₗ[R] V) {r : R}
    {x : HahnSeries Γ U} : (r • x).map f = r • ((x.map f) : HahnSeries Γ V) := by
  /-
    Γ : Type u_1
    R : Type u_3
    U : Type u_5
    V : Type u_6
    inst✝⁵ : PartialOrder Γ
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid V
    inst✝² : Module R V
    inst✝¹ : AddCommMonoid U
    inst✝ : Module R U
    f : LinearMap (RingHom.id R) U V
    r : R
    x : HahnSeries Γ U
    ⊢ Eq ((HSMul.hSMul r x).map f) (HSMul.hSMul r (x.map f))
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem embDomain_smul (f : Γ ↪o Γ') (r : R) (x : HahnSeries Γ R) :
    embDomain f (r • x) = r • embDomain f x := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Semiring R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    r : R
    x : HahnSeries Γ R
    ⊢ Eq (HahnSeries.embDomain f (HSMul.hSMul r x)) (HSMul.hSMul r (HahnSeries.emb …
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    inst✝² : PartialOrder Γ
    inst✝¹ : Semiring R
    inst✝ : PartialOrder Γ'
    f : OrderEmbedding Γ Γ'
    r : R
    x : HahnSeries Γ R
    g : Γ'
    ⊢ Eq ((HahnSeries.embDomain f (HSMul.hSMul r x)).coeff g) ((HSMul.hSMul r (Hah …
  -/
  by_cases hg : g ∈ Set.range f
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Semiring R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      r : R
      x : HahnSeries Γ R
      g : Γ'
      hg : Membership.mem (Set.range ⇑f) g
      ⊢ Eq ((HahnSeries.embDomain f (HSMul.hSMul r x)).coeff g) ((HSMul.hSMul r (Hah …
    -/
  · obtain ⟨a, rfl⟩ := hg
    /-
      case pos.intro
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Semiring R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      r : R
      x : HahnSeries Γ R
      a : Γ
      ⊢ Eq ((HahnSeries.embDomain f (HSMul.hSMul r x)).coeff (f a)) ((HSMul.hSMul r  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      inst✝² : PartialOrder Γ
      inst✝¹ : Semiring R
      inst✝ : PartialOrder Γ'
      f : OrderEmbedding Γ Γ'
      r : R
      x : HahnSeries Γ R
      g : Γ'
      hg : Not (Membership.mem (Set.range ⇑f) g)
      ⊢ Eq ((HahnSeries.embDomain f (HSMul.hSMul r x)).coeff g) ((HSMul.hSMul r (Hah …
    -/
  · simp [embDomain_notin_range hg]
    /-
      🎉 no goals
    -/


/-- Extending the domain of Hahn series is a linear map. -/
@[simps]
def embDomainLinearMap (f : Γ ↪o Γ') : HahnSeries Γ R →ₗ[R] HahnSeries Γ' R where
  toFun := embDomain f
  map_add' := embDomain_add f
  map_smul' := embDomain_smul f


