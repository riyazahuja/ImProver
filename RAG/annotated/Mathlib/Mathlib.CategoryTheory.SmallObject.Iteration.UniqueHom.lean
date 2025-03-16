/-- The (unique) morphism between two objects in `Iteration ε ⊥` -/
def mkOfBot (iter₁ iter₂ : Iteration ε (⊥ : J)) : iter₁ ⟶ iter₂ where
  natTrans :=
                                        /-
                                          C : Type u_1
                                          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
                                          Φ : CategoryTheory.Functor C C
                                          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                          J : Type u
                                          inst✝² : LinearOrder J
                                          inst✝¹ : OrderBot J
                                          inst✝ : SuccOrder J
                                          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
                                          x✝ : ↑(Set.Iic Bot.bot)
                                          i : J
                                          hi : Membership.mem (Set.Iic Bot.bot) i
                                          ⊢ Eq (iter₁.F.obj ⟨i, hi⟩) (iter₁.F.obj ⟨Bot.bot, ⋯⟩)
                                        -/
    { app := fun ⟨i, hi⟩ => eqToHom (by congr; simpa using hi) ≫ iter₁.isoZero.hom ≫
                                               /-
                                                 🎉 no goals
                                               -/
                                        /-
                                          C : Type u_1
                                          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
                                          Φ : CategoryTheory.Functor C C
                                          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                          J : Type u
                                          inst✝² : LinearOrder J
                                          inst✝¹ : OrderBot J
                                          inst✝ : SuccOrder J
                                          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
                                          x✝ : ↑(Set.Iic Bot.bot)
                                          i : J
                                          hi : Membership.mem (Set.Iic Bot.bot) i
                                          ⊢ Eq (iter₂.F.obj ⟨Bot.bot, ⋯⟩) (iter₂.F.obj ⟨i, hi⟩)
                                        -/
        iter₂.isoZero.inv ≫ eqToHom (by congr; symm; simpa using hi)
                                                     /-
                                                       🎉 no goals
                                                     -/
      naturality := fun ⟨i, hi⟩ ⟨k, hk⟩ φ => by
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
          x✝¹ x✝ : ↑(Set.Iic Bot.bot)
          i : J
          hi : Membership.mem (Set.Iic Bot.bot) i
          k : J
          hk : Membership.mem (Set.Iic Bot.bot) k
          φ : Quiver.Hom ⟨i, hi⟩ ⟨k, hk⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map φ) ((fun x => CategoryTh …
        -/
        obtain rfl : i = ⊥ := by simpa using hi
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
          x✝¹ x✝ : ↑(Set.Iic Bot.bot)
          k : J
          hk : Membership.mem (Set.Iic Bot.bot) k
          hi : Membership.mem (Set.Iic Bot.bot) Bot.bot
          φ : Quiver.Hom ⟨Bot.bot, hi⟩ ⟨k, hk⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map φ) ((fun x => CategoryTh …
        -/
        obtain rfl : k = ⊥ := by simpa using hk
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
          x✝¹ x✝ : ↑(Set.Iic Bot.bot)
          hi hk : Membership.mem (Set.Iic Bot.bot) Bot.bot
          φ : Quiver.Hom ⟨Bot.bot, hi⟩ ⟨Bot.bot, hk⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map φ) ((fun x => CategoryTh …
        -/
        obtain rfl : φ = 𝟙 _ := rfl
        /-
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
          x✝¹ x✝ : ↑(Set.Iic Bot.bot)
          hi hk : Membership.mem (Set.Iic Bot.bot) Bot.bot
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.Category …
        -/
        simp }
        /-
          🎉 no goals
        -/
                               /-
                                 C : Type u_1
                                 inst✝³ : CategoryTheory.Category.{?u.507, u_1} C
                                 Φ : CategoryTheory.Functor C C
                                 ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                 J : Type u
                                 inst✝² : LinearOrder J
                                 inst✝¹ : OrderBot J
                                 inst✝ : SuccOrder J
                                 iter₁ iter₂ : CategoryTheory.Functor.Iteration ε Bot.bot
                                 i : J
                                 hi : LT.lt i Bot.bot
                                 ⊢ Eq ({ app := fun x => CategoryTheory.Functor.Iteration.Hom.mkOfBot.match_1 ( …
                               -/
  natTrans_app_succ i hi := by simp at hi
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `mkOfSucc`. -/
noncomputable def mkOfSuccNatTransApp (k : J) (hk : k ≤ Order.succ i) :
    iter₁.F.obj ⟨k, hk⟩ ⟶ iter₂.F.obj ⟨k, hk⟩ :=
  if hk' : k = Order.succ i then
                /-
                  C : Type u_1
                  inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
                  Φ : CategoryTheory.Functor C C
                  ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                  J : Type u
                  inst✝² : LinearOrder J
                  inst✝¹ : OrderBot J
                  inst✝ : SuccOrder J
                  i : J
                  iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                  hi : Not (IsMax i)
                  φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                  k : J
                  hk : LE.le k (Order.succ i)
                  hk' : Eq k (Order.succ i)
                  ⊢ Eq (iter₁.F.obj ⟨k, hk⟩) (iter₁.F.obj ⟨Order.succ i, ⋯⟩)
                -/
    eqToHom (by subst hk'; rfl) ≫ (iter₁.isoSucc i (Order.lt_succ_of_not_isMax hi)).hom ≫
                           /-
                             🎉 no goals
                           -/
                                          /-
                                            C : Type u_1
                                            inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
                                            Φ : CategoryTheory.Functor C C
                                            ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                            J : Type u
                                            inst✝² : LinearOrder J
                                            inst✝¹ : OrderBot J
                                            inst✝ : SuccOrder J
                                            i : J
                                            iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                                            hi : Not (IsMax i)
                                            φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                            k : J
                                            hk : LE.le k (Order.succ i)
                                            hk' : Eq k (Order.succ i)
                                            ⊢ Membership.mem (Set.Iic i) i
                                          -/
      whiskerRight (φ.natTrans.app ⟨i, by simp⟩) _ ≫
                                          /-
                                            🎉 no goals
                                          -/
      (iter₂.isoSucc i (Order.lt_succ_of_not_isMax hi)).inv ≫
                  /-
                    C : Type u_1
                    inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
                    Φ : CategoryTheory.Functor C C
                    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                    J : Type u
                    inst✝² : LinearOrder J
                    inst✝¹ : OrderBot J
                    inst✝ : SuccOrder J
                    i : J
                    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                    hi : Not (IsMax i)
                    φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                    k : J
                    hk : LE.le k (Order.succ i)
                    hk' : Eq k (Order.succ i)
                    ⊢ Eq (iter₂.F.obj ⟨Order.succ i, ⋯⟩) (iter₂.F.obj ⟨k, hk⟩)
                  -/
      eqToHom (by subst hk'; rfl)
                             /-
                               🎉 no goals
                             -/
  else
    φ.natTrans.app ⟨k, Order.le_of_lt_succ (by
      /-
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        k : J
        hk : LE.le k (Order.succ i)
        hk' : Not (Eq k (Order.succ i))
        ⊢ LT.lt k (Order.succ i)
      -/
      obtain hk | rfl := hk.lt_or_eq
        /-
          case inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          k : J
          hk✝ : LE.le k (Order.succ i)
          hk' : Not (Eq k (Order.succ i))
          hk : LT.lt k (Order.succ i)
          ⊢ LT.lt k (Order.succ i)
        -/
      · exact hk
        /-
          🎉 no goals
        -/
        /-
          case inr
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.10543, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          hk : LE.le (Order.succ i) (Order.succ i)
          hk' : Not (Eq (Order.succ i) (Order.succ i))
          ⊢ LT.lt (Order.succ i) (Order.succ i)
        -/
      · tauto)⟩
        /-
          🎉 no goals
        -/


lemma mkOfSuccNatTransApp_eq_of_le (k : J) (hk : k ≤ i) :
    mkOfSuccNatTransApp hi φ k (hk.trans (Order.le_succ i)) =
      φ.natTrans.app ⟨k, hk⟩ :=
              /-
                C : Type u_1
                inst✝³ : CategoryTheory.Category.{u_2, u_1} C
                Φ : CategoryTheory.Functor C C
                ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                J : Type u
                inst✝² : LinearOrder J
                inst✝¹ : OrderBot J
                inst✝ : SuccOrder J
                i : J
                iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                hi : Not (IsMax i)
                φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                k : J
                hk : LE.le k i
                ⊢ Not (Eq k (Order.succ i))
              -/
  dif_neg (by rintro rfl; simpa using lt_of_le_of_lt hk (Order.lt_succ_of_not_isMax hi))
                          /-
                            🎉 no goals
                          -/


@[simp]
lemma mkOfSuccNatTransApp_succ_eq :
                                                /-
                                                  C : Type u_1
                                                  inst✝³ : CategoryTheory.Category.{?u.17075, u_1} C
                                                  Φ : CategoryTheory.Functor C C
                                                  ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                  J : Type u
                                                  inst✝² : LinearOrder J
                                                  inst✝¹ : OrderBot J
                                                  inst✝ : SuccOrder J
                                                  i : J
                                                  iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                                                  hi : Not (IsMax i)
                                                  φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                                  ⊢ LE.le (Order.succ i) (Order.succ i)
                                                -/
    mkOfSuccNatTransApp hi φ (Order.succ i) (by rfl) =
                                                /-
                                                  🎉 no goals
                                                -/
      (iter₁.isoSucc i (Order.lt_succ_of_not_isMax hi)).hom ≫
                                            /-
                                              C : Type u_1
                                              inst✝³ : CategoryTheory.Category.{?u.17075, u_1} C
                                              Φ : CategoryTheory.Functor C C
                                              ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                              J : Type u
                                              inst✝² : LinearOrder J
                                              inst✝¹ : OrderBot J
                                              inst✝ : SuccOrder J
                                              i : J
                                              iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
                                              hi : Not (IsMax i)
                                              φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                              ⊢ Membership.mem (Set.Iic i) i
                                            -/
        whiskerRight (φ.natTrans.app ⟨i, by simp⟩) _ ≫
                                            /-
                                              🎉 no goals
                                            -/
        (iter₂.isoSucc i (Order.lt_succ_of_not_isMax hi)).inv := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    i : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
    hi : Not (IsMax i)
    φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
    ⊢ Eq (CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTransApp hi φ (Order.suc …
  -/
  dsimp [mkOfSuccNatTransApp]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝² : LinearOrder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    i : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
    hi : Not (IsMax i)
    φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
    ⊢ Eq (dite (Eq (Order.succ i) (Order.succ i)) (fun hk' => CategoryTheory.Categ …
  -/
  rw [dif_pos rfl, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `mkOfSucc`. -/
@[simps]
noncomputable def mkOfSuccNatTrans :
    iter₁.F ⟶ iter₂.F where
  app := fun ⟨k, hk⟩ => mkOfSuccNatTransApp hi φ k hk
  naturality := fun ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩ f => by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
      k₁ : J
      hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
      k₂ : J
      hk₂ : Membership.mem (Set.Iic (Order.succ i)) k₂
      f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
      k₁ : J
      hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
      k₂ : J
      hk₂ : Membership.mem (Set.Iic (Order.succ i)) k₂
      f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
    -/
    have hk : k₁ ≤ k₂ := leOfHom f
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
      k₁ : J
      hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
      k₂ : J
      hk₂ : Membership.mem (Set.Iic (Order.succ i)) k₂
      f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
      hk : LE.le k₁ k₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
    -/
    obtain h₂ | rfl := hk₂.lt_or_eq
      /-
        case inl
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
        k₁ : J
        hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
        k₂ : J
        hk₂ : Membership.mem (Set.Iic (Order.succ i)) k₂
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
        hk : LE.le k₁ k₂
        h₂ : LT.lt k₂ (Order.succ i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
      -/
    · replace h₂ : k₂ ≤ i := Order.le_of_lt_succ h₂
      rw [mkOfSuccNatTransApp_eq_of_le hi φ k₂ h₂,
        mkOfSuccNatTransApp_eq_of_le hi φ k₁ (hk.trans h₂)]
      /-
        case inl
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
        k₁ : J
        hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
        k₂ : J
        hk₂ : Membership.mem (Set.Iic (Order.succ i)) k₂
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
        hk : LE.le k₁ k₂
        h₂ : LE.le k₂ i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (φ.natTrans.app ⟨k₂,  …
      -/
      exact natTrans_naturality φ k₁ k₂ hk h₂
      /-
        🎉 no goals
      -/
      /-
        case inr
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
        k₁ : J
        hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
        hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨Order.succ i, hk₂⟩
        hk : LE.le k₁ (Order.succ i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
      -/
    · obtain h₁ | rfl := hk.lt_or_eq
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨Order.succ i, hk₂⟩
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
        -/
      · have h₂ : k₁ ≤ i := Order.le_of_lt_succ h₁
        let f₁ : (⟨k₁, hk⟩ : { l | l ≤ Order.succ i}) ⟶
          ⟨i, Order.le_succ i⟩ := homOfLE h₂
        let f₂ : (⟨i, Order.le_succ i⟩ : Set.Iic (Order.succ i)) ⟶
          ⟨Order.succ i, by simp⟩ := homOfLE (Order.le_succ i)
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨Order.succ i, hk₂⟩
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
        -/
        obtain rfl : f = f₁ ≫ f₂ := rfl
        rw [Functor.map_comp, Functor.map_comp, assoc,
          mkOfSuccNatTransApp_eq_of_le hi φ k₁ h₂]
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        erw [← natTrans_naturality_assoc φ k₁ i h₂ (by rfl)]
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        rw [mkOfSuccNatTransApp_succ_eq]
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        dsimp
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        have ha : iter₁.F.map f₂ = iter₁.mapSucc i (Order.lt_succ_of_not_isMax hi) := rfl
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        have hb : iter₂.F.map f₂ = iter₂.mapSucc i (Order.lt_succ_of_not_isMax hi) := rfl
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          hb : Eq (iter₂.F.map f₂) (iter₂.mapSucc i ⋯)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        rw [ha, hb]
        rw [iter₁.mapSucc_eq i, iter₂.mapSucc_eq i, assoc,
          Iso.inv_hom_id_assoc]
        /-
          case inr.inl
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          hb : Eq (iter₂.F.map f₂) (iter₂.mapSucc i ⋯)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cate …
        -/
        ext X
        /-
          case inr.inl.w.h
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          hb : Eq (iter₂.F.map f₂) (iter₂.mapSucc i ⋯)
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (iter₁.F.map f₁) (CategoryTheory.Cat …
        -/
        dsimp
        /-
          case inr.inl.w.h
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          hb : Eq (iter₂.F.map f₂) (iter₂.mapSucc i ⋯)
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((iter₁.F.map f₁).app X) (CategoryThe …
        -/
        rw [← ε.naturality_assoc]
        /-
          case inr.inl.w.h
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          k₁ : J
          hk₁ : Membership.mem (Set.Iic (Order.succ i)) k₁
          hk₂ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le k₁ (Order.succ i)
          h₁ : LT.lt k₁ (Order.succ i)
          h₂ : LE.le k₁ i
          f₁ : Quiver.Hom ⟨k₁, hk⟩ ⟨i, ⋯⟩ := CategoryTheory.homOfLE h₂
          f₂ : Quiver.Hom ⟨i, ⋯⟩ ⟨Order.succ i, ⋯⟩ := CategoryTheory.homOfLE ⋯
          ha : Eq (iter₁.F.map f₂) (iter₁.mapSucc i ⋯)
          hb : Eq (iter₂.F.map f₂) (iter₂.mapSucc i ⋯)
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((iter₁.F.map f₁).app X) (CategoryThe …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          hk₂ hk₁ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          f : Quiver.Hom ⟨Order.succ i, hk₁⟩ ⟨Order.succ i, hk₂⟩
          hk : LE.le (Order.succ i) (Order.succ i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) (CategoryTheory.Funct …
        -/
      · obtain rfl : f = 𝟙 _ := rfl
        /-
          case inr.inr
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.20780, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝² : LinearOrder J
          inst✝¹ : OrderBot J
          inst✝ : SuccOrder J
          i : J
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
          hi : Not (IsMax i)
          φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          x✝¹ x✝ : ↑(Set.Iic (Order.succ i))
          hk₂ hk₁ : Membership.mem (Set.Iic (Order.succ i)) (Order.succ i)
          hk : LE.le (Order.succ i) (Order.succ i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.Category …
        -/
        rw [Functor.map_id, Functor.map_id, id_comp, comp_id]
        /-
          🎉 no goals
        -/


/-- The (unique) morphism between two objects in `Iteration ε (Order.succ i)`,
assuming we have a morphism between the truncations to `Iteration ε i`. -/
noncomputable def mkOfSucc
    {i : J} (iter₁ iter₂ : Iteration ε (Order.succ i)) (hi : ¬IsMax i)
    (φ : iter₁.trunc (SuccOrder.le_succ i) ⟶ iter₂.trunc (SuccOrder.le_succ i)) :
    iter₁ ⟶ iter₂ where
  natTrans := mkOfSuccNatTrans hi φ
  natTrans_app_zero := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTrans hi φ).app ⟨Bot.bo …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      ⊢ Eq (CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTransApp hi φ Bot.bot ⋯) …
    -/
    rw [mkOfSuccNatTransApp_eq_of_le _ _ _ bot_le, φ.natTrans_app_zero]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.trunc ⋯).isoZero.hom (iter₂.tr …
    -/
    rfl
    /-
      🎉 no goals
    -/
  natTrans_app_succ k hk := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝² : LinearOrder J
      inst✝¹ : OrderBot J
      inst✝ : SuccOrder J
      i : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
      hi : Not (IsMax i)
      φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      k : J
      hk : LT.lt k (Order.succ i)
      ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTrans hi φ).app ⟨Order. …
    -/
    obtain hk' | rfl := (Order.le_of_lt_succ hk).lt_or_eq
      /-
        case inl
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        k : J
        hk : LT.lt k (Order.succ i)
        hk' : LT.lt k i
        ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTrans hi φ).app ⟨Order. …
      -/
    · dsimp
      rw [mkOfSuccNatTransApp_eq_of_le hi φ k hk'.le,
        mkOfSuccNatTransApp_eq_of_le hi φ (Order.succ k) (Order.succ_le_of_lt hk'),
        φ.natTrans_app_succ _ hk']
      /-
        case inl
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        i : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ i)
        hi : Not (IsMax i)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        k : J
        hk : LT.lt k (Order.succ i)
        hk' : LT.lt k i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((iter₁.trunc ⋯).isoSucc k hk').hom ( …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case inr
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.33500, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝² : LinearOrder J
        inst✝¹ : OrderBot J
        inst✝ : SuccOrder J
        k : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε (Order.succ k)
        hi : Not (IsMax k)
        φ : Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        hk : LT.lt k (Order.succ k)
        ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfSuccNatTrans hi φ).app ⟨Order. …
      -/
    · simp [mkOfSuccNatTransApp_eq_of_le hi φ k (by rfl)]
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `mkOfLimit`. -/
def mkOfLimitNatTransApp (i : J) (hi : i ≤ j) :
    iter₁.F.obj ⟨i, hi⟩ ⟶ iter₂.F.obj ⟨i, hi⟩ :=
  if h : i < j
    then
                                  /-
                                    C : Type u_1
                                    inst✝⁴ : CategoryTheory.Category.{?u.40090, u_1} C
                                    Φ : CategoryTheory.Functor C C
                                    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                    J : Type u
                                    inst✝³ : LinearOrder J
                                    inst✝² : OrderBot J
                                    inst✝¹ : SuccOrder J
                                    j : J
                                    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                    φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                    inst✝ : WellFoundedLT J
                                    hj : Order.IsSuccLimit j
                                    i : J
                                    hi : LE.le i j
                                    h : LT.lt i j
                                    ⊢ Membership.mem (Set.Iic i) i
                                  -/
      (φ i h).natTrans.app ⟨i, by simp⟩
                                  /-
                                    🎉 no goals
                                  -/
    else by
      obtain rfl : i = j := by
        obtain h' | rfl := hi.lt_or_eq
        · exfalso
          exact h h'
        · rfl
      exact (iter₁.isColimit i hj (by simp)).desc (Cocone.mk _
        { app := fun ⟨k, hk⟩ => (φ k hk).natTrans.app ⟨k, by simp⟩ ≫
            iter₂.F.map (homOfLE (by exact hk.le))
          naturality := fun ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩ f => by
            have hf : k₁ ≤ k₂ := by simpa using leOfHom f
            dsimp
            rw [comp_id, congr_app (φ k₁ hk₁) ((truncFunctor ε hf).map (φ k₂ hk₂))]
            erw [natTrans_naturality_assoc (φ k₂ hk₂) k₁ k₂ hf (by rfl)]
            dsimp
            rw [← Functor.map_comp, homOfLE_comp] })


lemma mkOfLimitNatTransApp_eq_of_lt (i : J) (hi : i < j) :
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝⁴ : CategoryTheory.Category.{?u.53315, u_1} C
                                                                       Φ : CategoryTheory.Functor C C
                                                                       ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                       J : Type u
                                                                       inst✝³ : LinearOrder J
                                                                       inst✝² : OrderBot J
                                                                       inst✝¹ : SuccOrder J
                                                                       j : J
                                                                       iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                                                       φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                                                       inst✝ : WellFoundedLT J
                                                                       hj : Order.IsSuccLimit j
                                                                       i : J
                                                                       hi : LT.lt i j
                                                                       ⊢ Membership.mem (Set.Iic i) i
                                                                     -/
    mkOfLimitNatTransApp φ hj i hi.le = (φ i hi).natTrans.app ⟨i, by simp⟩ :=
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  dif_pos hi


lemma mkOfLimitNatTransApp_naturality_top (i : J) (hi : i < j) :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝⁴ : CategoryTheory.Category.{?u.55858, u_1} C
                                                                        Φ : CategoryTheory.Functor C C
                                                                        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                        J : Type u
                                                                        inst✝³ : LinearOrder J
                                                                        inst✝² : OrderBot J
                                                                        inst✝¹ : SuccOrder J
                                                                        j : J
                                                                        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                                                        φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                                                        inst✝ : WellFoundedLT J
                                                                        hj : Order.IsSuccLimit j
                                                                        i : J
                                                                        hi : LT.lt i j
                                                                        ⊢ Membership.mem (Set.Iic j) j
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    iter₁.F.map (homOfLE (by simpa using hi.le) : ⟨i, hi.le⟩ ⟶ ⟨j, by simp⟩) ≫
                             /-
                               🎉 no goals
                             -/
                                      /-
                                        C : Type u_1
                                        inst✝⁴ : CategoryTheory.Category.{?u.55858, u_1} C
                                        Φ : CategoryTheory.Functor C C
                                        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                        J : Type u
                                        inst✝³ : LinearOrder J
                                        inst✝² : OrderBot J
                                        inst✝¹ : SuccOrder J
                                        j : J
                                        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                        φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                        inst✝ : WellFoundedLT J
                                        hj : Order.IsSuccLimit j
                                        i : J
                                        hi : LT.lt i j
                                        ⊢ LE.le j j
                                      -/
      mkOfLimitNatTransApp φ hj j (by rfl) =
                                      /-
                                        🎉 no goals
                                      -/
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝⁴ : CategoryTheory.Category.{?u.55858, u_1} C
                                                                     Φ : CategoryTheory.Functor C C
                                                                     ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                     J : Type u
                                                                     inst✝³ : LinearOrder J
                                                                     inst✝² : OrderBot J
                                                                     inst✝¹ : SuccOrder J
                                                                     j : J
                                                                     iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                                                     φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
                                                                     inst✝ : WellFoundedLT J
                                                                     hj : Order.IsSuccLimit j
                                                                     i : J
                                                                     hi : LT.lt i j
                                                                     ⊢ LE.le ⟨i, ⋯⟩ ⟨j, ⋯⟩
                                                                   -/
      mkOfLimitNatTransApp φ hj i hi.le ≫ iter₂.F.map (homOfLE (by simpa using hi.le)) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    j : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
    inst✝ : WellFoundedLT J
    hj : Order.IsSuccLimit j
    i : J
    hi : LT.lt i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.homOfLE  …
  -/
  rw [mkOfLimitNatTransApp_eq_of_lt φ hj i hi, mkOfLimitNatTransApp, dif_neg (by simp)]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    j : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
    inst✝ : WellFoundedLT J
    hj : Order.IsSuccLimit j
    i : J
    hi : LT.lt i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.homOfLE  …
  -/
  exact (iter₁.isColimit j hj (by rfl)).fac _ ⟨i, hi⟩
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `mkOfLimit`. -/
@[simps]
def mkOfLimitNatTrans : iter₁.F ⟶ iter₂.F where
  app := fun ⟨k, hk⟩ => mkOfLimitNatTransApp φ hj k hk
  naturality := fun ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩ f => by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      x✝¹ x✝ : ↑(Set.Iic j)
      k₁ : J
      hk₁ : Membership.mem (Set.Iic j) k₁
      k₂ : J
      hk₂ : Membership.mem (Set.Iic j) k₂
      f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
    -/
    have hk : k₁ ≤ k₂ := leOfHom f
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      x✝¹ x✝ : ↑(Set.Iic j)
      k₁ : J
      hk₁ : Membership.mem (Set.Iic j) k₁
      k₂ : J
      hk₂ : Membership.mem (Set.Iic j) k₂
      f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
      hk : LE.le k₁ k₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
    -/
    obtain h₂ | rfl := hk₂.lt_or_eq
      /-
        case inl
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : OrderBot J
        inst✝¹ : SuccOrder J
        j : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
        φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        inst✝ : WellFoundedLT J
        hj : Order.IsSuccLimit j
        x✝¹ x✝ : ↑(Set.Iic j)
        k₁ : J
        hk₁ : Membership.mem (Set.Iic j) k₁
        k₂ : J
        hk₂ : Membership.mem (Set.Iic j) k₂
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
        hk : LE.le k₁ k₂
        h₂ : LT.lt k₂ j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
      -/
    · dsimp
      rw [mkOfLimitNatTransApp_eq_of_lt _ hj k₂ h₂,
        mkOfLimitNatTransApp_eq_of_lt _ hj k₁ (lt_of_le_of_lt hk h₂),
        congr_app (φ k₁ (lt_of_le_of_lt hk h₂)) ((truncFunctor ε hk).map (φ k₂ h₂))]
      /-
        case inl
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : OrderBot J
        inst✝¹ : SuccOrder J
        j : J
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
        φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        inst✝ : WellFoundedLT J
        hj : Order.IsSuccLimit j
        x✝¹ x✝ : ↑(Set.Iic j)
        k₁ : J
        hk₁ : Membership.mem (Set.Iic j) k₁
        k₂ : J
        hk₂ : Membership.mem (Set.Iic j) k₂
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
        hk : LE.le k₁ k₂
        h₂ : LT.lt k₂ j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((φ k₂ h₂).natTrans.a …
      -/
      exact natTrans_naturality (φ k₂ h₂) k₁ k₂ hk (by rfl)
      /-
        🎉 no goals
      -/
      /-
        case inr
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
        Φ : CategoryTheory.Functor C C
        ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
        J : Type u
        inst✝³ : LinearOrder J
        inst✝² : OrderBot J
        inst✝¹ : SuccOrder J
        inst✝ : WellFoundedLT J
        k₁ k₂ : J
        hk : LE.le k₁ k₂
        iter₁ iter₂ : CategoryTheory.Functor.Iteration ε k₂
        φ : (i : J) → (hi : LT.lt i k₂) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
        hj : Order.IsSuccLimit k₂
        x✝¹ x✝ : ↑(Set.Iic k₂)
        hk₁ : Membership.mem (Set.Iic k₂) k₁
        hk₂ : Membership.mem (Set.Iic k₂) k₂
        f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
      -/
    · obtain h₁ | rfl := hk₁.lt_or_eq
        /-
          case inr.inl
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝³ : LinearOrder J
          inst✝² : OrderBot J
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          k₁ k₂ : J
          hk : LE.le k₁ k₂
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε k₂
          φ : (i : J) → (hi : LT.lt i k₂) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          hj : Order.IsSuccLimit k₂
          x✝¹ x✝ : ↑(Set.Iic k₂)
          hk₁ : Membership.mem (Set.Iic k₂) k₁
          hk₂ : Membership.mem (Set.Iic k₂) k₂
          f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
          h₁ : LT.lt k₁ k₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
        -/
      · exact mkOfLimitNatTransApp_naturality_top _ hj _ h₁
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝³ : LinearOrder J
          inst✝² : OrderBot J
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          k₁ : J
          hk : LE.le k₁ k₁
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε k₁
          φ : (i : J) → (hi : LT.lt i k₁) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          hj : Order.IsSuccLimit k₁
          x✝¹ x✝ : ↑(Set.Iic k₁)
          hk₁ hk₂ : Membership.mem (Set.Iic k₁) k₁
          f : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₁, hk₂⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map f) ((fun x => CategoryTh …
        -/
      · obtain rfl : f = 𝟙 _ := rfl
        /-
          case inr.inr
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{?u.60931, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝³ : LinearOrder J
          inst✝² : OrderBot J
          inst✝¹ : SuccOrder J
          inst✝ : WellFoundedLT J
          k₁ : J
          hk : LE.le k₁ k₁
          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε k₁
          φ : (i : J) → (hi : LT.lt i k₁) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
          hj : Order.IsSuccLimit k₁
          x✝¹ x✝ : ↑(Set.Iic k₁)
          hk₁ hk₂ : Membership.mem (Set.Iic k₁) k₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.Category …
        -/
        simp only [map_id, id_comp, comp_id]
        /-
          🎉 no goals
        -/


/-- The (unique) morphism between two objects in `Iteration ε j` when `j` is a limit element,
assuming we have a morphism between the truncations to `Iteration ε i` for all `i < j`. -/
def mkOfLimit : iter₁ ⟶ iter₂ where
  natTrans := mkOfLimitNatTrans φ hj
  natTrans_app_zero := by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.67497, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfLimitNatTrans φ hj).app ⟨Bot.b …
    -/
    simp [mkOfLimitNatTransApp_eq_of_lt φ _ ⊥ (by simpa only [bot_lt_iff_ne_bot] using hj.ne_bot)]
    /-
      🎉 no goals
    -/
  natTrans_app_succ i h := by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.67497, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      i : J
      h : LT.lt i j
      ⊢ Eq ((CategoryTheory.Functor.Iteration.Hom.mkOfLimitNatTrans φ hj).app ⟨Order …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.67497, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      i : J
      h : LT.lt i j
      ⊢ Eq (CategoryTheory.Functor.Iteration.Hom.mkOfLimitNatTransApp φ hj (Order.su …
    -/
    have h' := hj.succ_lt h
    rw [mkOfLimitNatTransApp_eq_of_lt φ hj _ h',
      (φ _ h').natTrans_app_succ i (Order.lt_succ_of_not_isMax (not_isMax_of_lt h)),
      mkOfLimitNatTransApp_eq_of_lt φ _ _ h,
      congr_app (φ i h) ((truncFunctor ε (Order.le_succ i)).map (φ (Order.succ i) h'))]
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.67497, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J : Type u
      inst✝³ : LinearOrder J
      inst✝² : OrderBot J
      inst✝¹ : SuccOrder J
      j : J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      φ : (i : J) → (hi : LT.lt i j) → Quiver.Hom (iter₁.trunc ⋯) (iter₂.trunc ⋯)
      inst✝ : WellFoundedLT J
      hj : Order.IsSuccLimit j
      i : J
      h : LT.lt i j
      h' : LT.lt (Order.succ i) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((iter₁.trunc ⋯).isoSucc i ⋯).hom (Ca …
    -/
    dsimp
    /-
      🎉 no goals
    -/


instance : Nonempty (iter₁ ⟶ iter₂) := by
  let P := fun (i : J) => ∀ (hi : i ≤ j),
    Nonempty ((truncFunctor ε hi).obj iter₁ ⟶ (truncFunctor ε hi).obj iter₂)
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    j : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    inst✝ : WellFoundedLT J
    P : J → Prop := fun i => ∀ (hi : LE.le i j), Nonempty (Quiver.Hom ((CategoryTh …
    ⊢ Nonempty (Quiver.Hom iter₁ iter₂)
  -/
  suffices ∀ i, P i from this j (by rfl)
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    j : J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    inst✝ : WellFoundedLT J
    P : J → Prop := fun i => ∀ (hi : LE.le i j), Nonempty (Quiver.Hom ((CategoryTh …
    ⊢ ∀ (i : J), P i
  -/
  intro i
  induction i using SuccOrder.limitRecOn with
  | hm i hi =>
    obtain rfl : i = ⊥ := by simpa using hi
    exact fun hi' ↦ ⟨Hom.mkOfBot _ _⟩
  | hs i hi hi' =>
    exact fun hi'' ↦ ⟨Hom.mkOfSucc _ _ hi (hi' ((Order.le_succ i).trans hi'')).some⟩
  | hl i hi hi' =>
    exact fun hi'' ↦ ⟨Hom.mkOfLimit (fun k hk ↦ (hi' k hk (hk.le.trans hi'')).some) hi⟩


noncomputable instance : Unique (iter₁ ⟶ iter₂) :=
  uniqueOfSubsingleton (Nonempty.some inferInstance)


/-- The canonical isomorphism between two objects in the category `Iteration ε j`. -/
noncomputable def iso : iter₁ ≅ iter₂ where
  hom := default
  inv := default


@[simp]
                                                    /-
                                                      C : Type u_1
                                                      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                      Φ : CategoryTheory.Functor C C
                                                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                      J : Type u
                                                      inst✝³ : LinearOrder J
                                                      inst✝² : OrderBot J
                                                      inst✝¹ : SuccOrder J
                                                      inst✝ : WellFoundedLT J
                                                      j : J
                                                      iter₁ : CategoryTheory.Functor.Iteration ε j
                                                      ⊢ Eq (iter₁.iso iter₁) (CategoryTheory.Iso.refl iter₁)
                                                    -/
lemma iso_refl : iso iter₁ iter₁ = Iso.refl _ := by aesop_cat
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                               Φ : CategoryTheory.Functor C C
                                                                               ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                                               J : Type u
                                                                               inst✝³ : LinearOrder J
                                                                               inst✝² : OrderBot J
                                                                               inst✝¹ : SuccOrder J
                                                                               inst✝ : WellFoundedLT J
                                                                               j : J
                                                                               iter₁ iter₂ iter₃ : CategoryTheory.Functor.Iteration ε j
                                                                               ⊢ Eq ((iter₁.iso iter₂).trans (iter₂.iso iter₃)) (iter₁.iso iter₃)
                                                                             -/
lemma iso_trans : iso iter₁ iter₂ ≪≫ iso iter₂ iter₃ = iso iter₁ iter₃ := by aesop_cat
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


