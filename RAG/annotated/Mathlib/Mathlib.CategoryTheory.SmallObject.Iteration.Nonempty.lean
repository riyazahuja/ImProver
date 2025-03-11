variable (ε J) in
/-- The obvious term in `Iteration ε ⊥`: it is given by the identity functor. -/
def mkOfBot : Iteration ε (⊥ : J) where
  F := (Functor.const _).obj (𝟭 C)
  isoZero := Iso.refl _
                    /-
                      C : Type u_1
                      inst✝³ : CategoryTheory.Category.{?u.1008, u_1} C
                      Φ : CategoryTheory.Functor C C
                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                      J : Type u
                      inst✝² : LinearOrder J
                      inst✝¹ : OrderBot J
                      inst✝ : SuccOrder J
                      x✝ : J
                      h : LT.lt x✝ Bot.bot
                      ⊢ CategoryTheory.Iso (((CategoryTheory.Functor.const ↑(Set.Iic Bot.bot)).obj ( …
                    -/
  isoSucc _ h := by simp at h
                    /-
                      🎉 no goals
                    -/
                        /-
                          C : Type u_1
                          inst✝³ : CategoryTheory.Category.{?u.1008, u_1} C
                          Φ : CategoryTheory.Functor C C
                          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                          J : Type u
                          inst✝² : LinearOrder J
                          inst✝¹ : OrderBot J
                          inst✝ : SuccOrder J
                          x✝ : J
                          h : LT.lt x✝ Bot.bot
                          ⊢ Eq (CategoryTheory.Functor.Iteration.mapSucc' ((CategoryTheory.Functor.const …
                        -/
  mapSucc'_eq _ h := by simp at h
                        /-
                          🎉 no goals
                        -/
  isColimit x hx h := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.1008, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      x : J
      hx : Order.IsSuccLimit x
      h : LE.le x Bot.bot
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Functor.Iteration.coconeOfLE …
    -/
    exfalso
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.1008, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      x : J
      hx : Order.IsSuccLimit x
      h : LE.le x Bot.bot
      ⊢ False
    -/
    refine hx.not_isMin (by simpa using h)
    /-
      🎉 no goals
    -/


/-- When `j : J` is not maximal, this is the extension as `Iteration ε (Order.succ j)`
of any `iter : Iteration ε j`. -/
noncomputable def mkOfSucc {j : J} (hj : ¬IsMax j) (iter : Iteration ε j) :
    Iteration ε (Order.succ j) where
  F := extendToSucc hj iter.F (whiskerLeft _ ε)
                                                                    /-
                                                                      C : Type u_1
                                                                      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
                                                                      Φ : CategoryTheory.Functor C C
                                                                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                      J : Type u
                                                                      inst✝² : LinearOrder J
                                                                      inst✝¹ : OrderBot J
                                                                      inst✝ : SuccOrder J
                                                                      j : J
                                                                      hj : Not (IsMax j)
                                                                      iter : CategoryTheory.Functor.Iteration ε j
                                                                      ⊢ Membership.mem (Set.Iic j) Bot.bot
                                                                    -/
  isoZero := (extendToSuccObjIso hj iter.F (whiskerLeft _ ε) ⟨⊥, by simp⟩).trans iter.isoZero
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  isoSucc i hi :=
    if hij : i < j then
      extendToSuccObjIso _ _ _ ⟨Order.succ i, Order.succ_le_of_lt hij⟩ ≪≫
        iter.isoSucc i hij ≪≫ (isoWhiskerRight (extendToSuccObjIso _ _ _ ⟨i, hij.le⟩).symm _)
    else
      have hij' : i = j := le_antisymm
            /-
              C : Type u_1
              inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
              Φ : CategoryTheory.Functor C C
              ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
              J : Type u
              inst✝² : LinearOrder J
              inst✝¹ : OrderBot J
              inst✝ : SuccOrder J
              j : J
              hj : Not (IsMax j)
              iter : CategoryTheory.Functor.Iteration ε j
              i : J
              hi : LT.lt i (Order.succ j)
              hij : Not (LT.lt i j)
              ⊢ LE.le i j
            -/
            /-
              🎉 no goals
            -/
        (by simpa only [Order.lt_succ_iff_of_not_isMax hj] using hi) (by simpa using hij)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
                  /-
                    C : Type u_1
                    inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
                    Φ : CategoryTheory.Functor C C
                    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                    J : Type u
                    inst✝² : LinearOrder J
                    inst✝¹ : OrderBot J
                    inst✝ : SuccOrder J
                    j : J
                    hj : Not (IsMax j)
                    iter : CategoryTheory.Functor.Iteration ε j
                    i : J
                    hi : LT.lt i (Order.succ j)
                    hij : Not (LT.lt i j)
                    hij' : Eq i j
                    ⊢ Eq ((CategoryTheory.Functor.extendToSucc hj iter.F (CategoryTheory.whiskerLe …
                  -/
      eqToIso (by subst hij'; rfl) ≪≫ extendToSuccObjSuccIso hj iter.F (whiskerLeft _ ε) ≪≫
                              /-
                                🎉 no goals
                              -/
                                                                                /-
                                                                                  C : Type u_1
                                                                                  inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
                                                                                  Φ : CategoryTheory.Functor C C
                                                                                  ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                                  J : Type u
                                                                                  inst✝² : LinearOrder J
                                                                                  inst✝¹ : OrderBot J
                                                                                  inst✝ : SuccOrder J
                                                                                  j : J
                                                                                  hj : Not (IsMax j)
                                                                                  iter : CategoryTheory.Functor.Iteration ε j
                                                                                  i : J
                                                                                  hi : LT.lt i (Order.succ j)
                                                                                  hij : Not (LT.lt i j)
                                                                                  hij' : Eq i j
                                                                                  ⊢ Membership.mem (Set.Iic j) j
                                                                                -/
        isoWhiskerRight ((extendToSuccObjIso hj iter.F (whiskerLeft _ ε) ⟨j, by simp⟩).symm.trans
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                         /-
                           C : Type u_1
                           inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
                           Φ : CategoryTheory.Functor C C
                           ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                           J : Type u
                           inst✝² : LinearOrder J
                           inst✝¹ : OrderBot J
                           inst✝ : SuccOrder J
                           j : J
                           hj : Not (IsMax j)
                           iter : CategoryTheory.Functor.Iteration ε j
                           i : J
                           hi : LT.lt i (Order.succ j)
                           hij : Not (LT.lt i j)
                           hij' : Eq i j
                           ⊢ Eq ((CategoryTheory.Functor.extendToSucc hj iter.F (CategoryTheory.whiskerLe …
                         -/
            (eqToIso (by subst hij'; rfl))) _
                                     /-
                                       🎉 no goals
                                     -/
  mapSucc'_eq i hi := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      iter : CategoryTheory.Functor.Iteration ε j
      i : J
      hi : LT.lt i (Order.succ j)
      ⊢ Eq (CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.extend …
    -/
    obtain hi' | rfl := ((Order.lt_succ_iff_of_not_isMax hj).mp hi).lt_or_eq
      /-
        case inl
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        ⊢ Eq (CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.extend …
      -/
    · ext X
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        ⊢ Eq ((CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.exten …
      -/
      have := iter.mapSucc_eq i hi'
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        this : Eq (iter.mapSucc i hi') (CategoryTheory.CategoryStruct.comp (CategoryTh …
        ⊢ Eq ((CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.exten …
      -/
      dsimp [mapSucc, mapSucc'] at this ⊢
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        this : Eq (iter.F.map (CategoryTheory.homOfLE ⋯)) (CategoryTheory.CategoryStru …
        ⊢ Eq (((CategoryTheory.Functor.extendToSucc hj iter.F (CategoryTheory.whiskerL …
      -/
      rw [extentToSucc_map _ _ _ _ _ _ (Order.succ_le_of_lt hi'), this, dif_pos hi']
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        this : Eq (iter.F.map (CategoryTheory.homOfLE ⋯)) (CategoryTheory.CategoryStru …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.extendToSucc …
      -/
      dsimp
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        this : Eq (iter.F.map (CategoryTheory.homOfLE ⋯)) (CategoryTheory.CategoryStru …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.extendToSucc …
      -/
      rw [assoc, assoc]
      /-
        case inl.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        iter : CategoryTheory.Functor.Iteration ε j
        i : J
        hi : LT.lt i (Order.succ j)
        hi' : LT.lt i j
        X : C
        this : Eq (iter.F.map (CategoryTheory.homOfLE ⋯)) (CategoryTheory.CategoryStru …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.extendToSucc …
      -/
      erw [ε.naturality_assoc]
      /-
        🎉 no goals
      -/
      /-
        case inr
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        ⊢ Eq (CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.extend …
      -/
    · ext X
      /-
        case inr.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        X : C
        ⊢ Eq ((CategoryTheory.Functor.Iteration.mapSucc' (CategoryTheory.Functor.exten …
      -/
      dsimp [mapSucc']
      /-
        case inr.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        X : C
        ⊢ Eq (((CategoryTheory.Functor.extendToSucc hj iter.F (CategoryTheory.whiskerL …
      -/
      rw [dif_neg (gt_irrefl i), extendToSucc_map_le_succ]
      /-
        case inr.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        X : C
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.extendToSucc …
      -/
      dsimp
      /-
        case inr.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.extendToSucc …
      -/
      rw [id_comp, comp_id]
      /-
        case inr.w.h
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        hj : Not (IsMax i)
        iter : CategoryTheory.Functor.Iteration ε i
        hi : LT.lt i (Order.succ i)
        X : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.extendToSucc …
      -/
      erw [ε.naturality_assoc]
      /-
        🎉 no goals
      -/
  isColimit i hi hij := by
    have hij' : i ≤ j := by
      obtain hij | rfl := hij.lt_or_eq
      · exact (Order.lt_succ_iff_of_not_isMax hj).1 hij
      · exfalso
        exact Order.not_isSuccLimit_succ_of_not_isMax hj hi
    refine (IsColimit.precomposeHomEquiv
      (isoWhiskerLeft (monotone_inclusion_lt_le_of_le hij').functor
        (extendToSuccRestrictionLEIso hj iter.F (whiskerLeft _ ε))).symm _).1
      (IsColimit.ofIsoColimit (iter.isColimit i hi hij')
      (Iso.symm (Cocones.ext (extendToSuccObjIso hj iter.F (whiskerLeft _ ε) ⟨i, hij'⟩)
      (fun ⟨k, hk⟩ ↦ ?_))))
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      iter : CategoryTheory.Functor.Iteration ε j
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i (Order.succ j)
      hij' : LE.le i j
      x✝ : Subtype fun i_1 => LT.lt i_1 i
      k : J
      hk : LT.lt k i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      iter : CategoryTheory.Functor.Iteration ε j
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i (Order.succ j)
      hij' : LE.le i j
      x✝ : Subtype fun i_1 => LT.lt i_1 i
      k : J
      hk : LT.lt k i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, extendToSuccObjIso_hom_naturality hj iter.F (whiskerLeft _ ε)]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      iter : CategoryTheory.Functor.Iteration ε j
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i (Order.succ j)
      hij' : LE.le i j
      x✝ : Subtype fun i_1 => LT.lt i_1 i
      k : J
      hk : LT.lt k i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.extendToSuccO …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.3882, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      iter : CategoryTheory.Functor.Iteration ε j
      i : J
      hi : Order.IsSuccLimit i
      hij : LE.le i (Order.succ j)
      hij' : LE.le i j
      x✝ : Subtype fun i_1 => LT.lt i_1 i
      k : J
      hk : LT.lt k i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.extendToSuccO …
    -/
    rw [Iso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/


