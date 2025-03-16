/-- A function to create a provable equal copy of a top order
with possibly different definitional equalities. -/
def OrderTop.copy {h : LE α} {h' : LE α} (c : @OrderTop α h')
                                  /-
                                    α : Type u
                                    h h' : LE α
                                    c : OrderTop α
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
    (le_eq : ∀ x y : α, (@LE.le α h) x y ↔ x ≤ y) : @OrderTop α h :=
                                             /-
                                               α : Type u
                                               h h' : LE α
                                               c : OrderTop α
                                               top : α
                                               eq_top : Eq top Top.top
                                               le_eq : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                               x✝ : α
                                               ⊢ LE.le x✝ Top.top
                                             -/
  @OrderTop.mk α h { top := top } fun _ ↦ by simp [eq_top, le_eq]
                                             /-
                                               🎉 no goals
                                             -/


/-- A function to create a provable equal copy of a bottom order
with possibly different definitional equalities. -/
def OrderBot.copy {h : LE α} {h' : LE α} (c : @OrderBot α h')
                                  /-
                                    α : Type u
                                    h h' : LE α
                                    c : OrderBot α
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
    (le_eq : ∀ x y : α, (@LE.le α h) x y ↔ x ≤ y) : @OrderBot α h :=
                                             /-
                                               α : Type u
                                               h h' : LE α
                                               c : OrderBot α
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               le_eq : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                               x✝ : α
                                               ⊢ LE.le Bot.bot x✝
                                             -/
  @OrderBot.mk α h { bot := bot } fun _ ↦ by simp [eq_bot, le_eq]
                                             /-
                                               🎉 no goals
                                             -/


/-- A function to create a provable equal copy of a bounded order
with possibly different definitional equalities. -/
def BoundedOrder.copy {h : LE α} {h' : LE α} (c : @BoundedOrder α h')
                                  /-
                                    α : Type u
                                    h h' : LE α
                                    c : BoundedOrder α
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    h h' : LE α
                                    c : BoundedOrder α
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
    (le_eq : ∀ x y : α, (@LE.le α h) x y ↔ x ≤ y) : @BoundedOrder α h :=
                                                                    /-
                                                                      α : Type u
                                                                      h h' : LE α
                                                                      c : BoundedOrder α
                                                                      top : α
                                                                      eq_top : Eq top Top.top
                                                                      bot : α
                                                                      eq_bot : Eq bot Bot.bot
                                                                      le_eq : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                                                      x✝ : α
                                                                      ⊢ LE.le x✝ Top.top
                                                                    -/
  @BoundedOrder.mk α h (@OrderTop.mk α h { top := top } (fun _ ↦ by simp [eq_top, le_eq]))
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                 /-
                                                   α : Type u
                                                   h h' : LE α
                                                   c : BoundedOrder α
                                                   top : α
                                                   eq_top : Eq top Top.top
                                                   bot : α
                                                   eq_bot : Eq bot Bot.bot
                                                   le_eq : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
                                                   x✝ : α
                                                   ⊢ LE.le Bot.bot x✝
                                                 -/
    (@OrderBot.mk α h { bot := bot } (fun _ ↦ by simp [eq_bot, le_eq]))
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A function to create a provable equal copy of a lattice
with possibly different definitional equalities. -/
def Lattice.copy (c : Lattice α)
                                          /-
                                            α : Type u
                                            c : Lattice α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : Lattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : Lattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min) : Lattice α where
                                          /-
                                            🎉 no goals
                                          -/
  le := le
  sup := sup
  inf := inf
  lt := fun a b ↦ le a b ∧ ¬ le b a
                /-
                  α : Type u
                  c : Lattice α
                  le : α → α → Prop
                  eq_le : Eq le LE.le
                  sup : α → α → α
                  eq_sup : Eq sup Max.max
                  inf : α → α → α
                  eq_inf : Eq inf Min.min
                  ⊢ ∀ (a : α), LE.le a a
                -/
  le_refl := by intros; simp [eq_le]
                        /-
                          🎉 no goals
                        -/
                 /-
                   α : Type u
                   c : Lattice α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   ⊢ ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
                 -/
  le_trans := by intro _ _ _ hab hbc; rw [eq_le] at hab hbc ⊢; exact le_trans hab hbc
                                                               /-
                                                                 🎉 no goals
                                                               -/
                    /-
                      α : Type u
                      c : Lattice α
                      le : α → α → Prop
                      eq_le : Eq le LE.le
                      sup : α → α → α
                      eq_sup : Eq sup Max.max
                      inf : α → α → α
                      eq_inf : Eq inf Min.min
                      ⊢ ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
                    -/
  le_antisymm := by intro _ _ hab hba; simp_rw [eq_le] at hab hba; exact le_antisymm hab hba
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                    /-
                      α : Type u
                      c : Lattice α
                      le : α → α → Prop
                      eq_le : Eq le LE.le
                      sup : α → α → α
                      eq_sup : Eq sup Max.max
                      inf : α → α → α
                      eq_inf : Eq inf Min.min
                      ⊢ ∀ (a b : α), LE.le a (sup a b)
                    -/
  le_sup_left := by intros; simp [eq_le, eq_sup]
                            /-
                              🎉 no goals
                            -/
                     /-
                       α : Type u
                       c : Lattice α
                       le : α → α → Prop
                       eq_le : Eq le LE.le
                       sup : α → α → α
                       eq_sup : Eq sup Max.max
                       inf : α → α → α
                       eq_inf : Eq inf Min.min
                       ⊢ ∀ (a b : α), LE.le b (sup a b)
                     -/
  le_sup_right := by intros; simp [eq_le, eq_sup]
                             /-
                               🎉 no goals
                             -/
               /-
                 α : Type u
                 c : Lattice α
                 le : α → α → Prop
                 eq_le : Eq le LE.le
                 sup : α → α → α
                 eq_sup : Eq sup Max.max
                 inf : α → α → α
                 eq_inf : Eq inf Min.min
                 ⊢ ∀ (a b c_1 : α), LE.le a c_1 → LE.le b c_1 → LE.le (sup a b) c_1
               -/
  sup_le := by intro _ _ _ hac hbc; simp_rw [eq_le] at hac hbc ⊢; simp [eq_sup, hac, hbc]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                    /-
                      α : Type u
                      c : Lattice α
                      le : α → α → Prop
                      eq_le : Eq le LE.le
                      sup : α → α → α
                      eq_sup : Eq sup Max.max
                      inf : α → α → α
                      eq_inf : Eq inf Min.min
                      ⊢ ∀ (a b : α), LE.le (inf a b) a
                    -/
  inf_le_left := by intros; simp [eq_le, eq_inf]
                            /-
                              🎉 no goals
                            -/
                     /-
                       α : Type u
                       c : Lattice α
                       le : α → α → Prop
                       eq_le : Eq le LE.le
                       sup : α → α → α
                       eq_sup : Eq sup Max.max
                       inf : α → α → α
                       eq_inf : Eq inf Min.min
                       ⊢ ∀ (a b : α), LE.le (inf a b) b
                     -/
  inf_le_right := by intros; simp [eq_le, eq_inf]
                             /-
                               🎉 no goals
                             -/
               /-
                 α : Type u
                 c : Lattice α
                 le : α → α → Prop
                 eq_le : Eq le LE.le
                 sup : α → α → α
                 eq_sup : Eq sup Max.max
                 inf : α → α → α
                 eq_inf : Eq inf Min.min
                 ⊢ ∀ (a b c_1 : α), LE.le a b → LE.le a c_1 → LE.le a (inf b c_1)
               -/
  le_inf := by intro _ _ _ hac hbc; simp_rw [eq_le] at hac hbc ⊢; simp [eq_inf, hac, hbc]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A function to create a provable equal copy of a distributive lattice
with possibly different definitional equalities. -/
def DistribLattice.copy (c : DistribLattice α)
                                          /-
                                            α : Type u
                                            c : DistribLattice α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : DistribLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : DistribLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min) : DistribLattice α where
                                          /-
                                            🎉 no goals
                                          -/
  toLattice := Lattice.copy (@DistribLattice.toLattice α c) le eq_le sup eq_sup inf eq_inf
                   /-
                     α : Type u
                     c : DistribLattice α
                     le : α → α → Prop
                     eq_le : Eq le LE.le
                     sup : α → α → α
                     eq_sup : Eq sup Max.max
                     inf : α → α → α
                     eq_inf : Eq inf Min.min
                     ⊢ ∀ (x y z : α), LE.le (Min.min (Max.max x y) (Max.max x z)) (Max.max x (Min.m …
                   -/
  le_sup_inf := by intros; simp [eq_le, eq_sup, eq_inf, le_sup_inf]
                           /-
                             🎉 no goals
                           -/


/-- A function to create a provable equal copy of a generalised heyting algebra
with possibly different definitional equalities. -/
def GeneralizedHeytingAlgebra.copy (c : GeneralizedHeytingAlgebra α)
                                          /-
                                            α : Type u
                                            c : GeneralizedHeytingAlgebra α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : GeneralizedHeytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : GeneralizedHeytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : GeneralizedHeytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                             /-
                                               α : Type u
                                               c : GeneralizedHeytingAlgebra α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               himp : α → α → α
                                               ⊢ HImp α
                                             -/
    (himp : α → α → α) (eq_himp : himp = (by infer_instance : HImp α).himp) :
                                             /-
                                               🎉 no goals
                                             -/
    GeneralizedHeytingAlgebra α where
  __ := Lattice.copy (@GeneralizedHeytingAlgebra.toLattice α c) le eq_le sup eq_sup inf eq_inf
  __ := OrderTop.copy (@GeneralizedHeytingAlgebra.toOrderTop α c) top eq_top
        /-
          α : Type u
          c : GeneralizedHeytingAlgebra α
          le : α → α → Prop
          eq_le : Eq le LE.le
          top : α
          eq_top : Eq top Top.top
          sup : α → α → α
          eq_sup : Eq sup Max.max
          inf : α → α → α
          eq_inf : Eq inf Min.min
          himp : α → α → α
          eq_himp : Eq himp HImp.himp
          ⊢ ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
        -/
    (by rw [← eq_le]; exact fun _ _ ↦ .rfl)
                      /-
                        🎉 no goals
                      -/
  himp := himp
                          /-
                            α : Type u
                            c : GeneralizedHeytingAlgebra α
                            le : α → α → Prop
                            eq_le : Eq le LE.le
                            top : α
                            eq_top : Eq top Top.top
                            sup : α → α → α
                            eq_sup : Eq sup Max.max
                            inf : α → α → α
                            eq_inf : Eq inf Min.min
                            himp : α → α → α
                            eq_himp : Eq himp HImp.himp
                            x✝² x✝¹ x✝ : α
                            ⊢ Iff (LE.le x✝² (HImp.himp x✝¹ x✝)) (LE.le (Min.min x✝² x✝¹) x✝)
                          -/
  le_himp_iff _ _ _ := by simp [eq_le, eq_himp, eq_inf]
                          /-
                            🎉 no goals
                          -/


/-- A function to create a provable equal copy of a generalised coheyting algebra
with possibly different definitional equalities. -/
def GeneralizedCoheytingAlgebra.copy (c : GeneralizedCoheytingAlgebra α)
                                          /-
                                            α : Type u
                                            c : GeneralizedCoheytingAlgebra α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : GeneralizedCoheytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : GeneralizedCoheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : GeneralizedCoheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                                /-
                                                  α : Type u
                                                  c : GeneralizedCoheytingAlgebra α
                                                  le : α → α → Prop
                                                  eq_le : Eq le LE.le
                                                  bot : α
                                                  eq_bot : Eq bot Bot.bot
                                                  sup : α → α → α
                                                  eq_sup : Eq sup Max.max
                                                  inf : α → α → α
                                                  eq_inf : Eq inf Min.min
                                                  sdiff : α → α → α
                                                  ⊢ SDiff α
                                                -/
    (sdiff : α → α → α) (eq_sdiff : sdiff = (by infer_instance : SDiff α).sdiff) :
                                                /-
                                                  🎉 no goals
                                                -/
    GeneralizedCoheytingAlgebra α where
  __ := Lattice.copy (@GeneralizedCoheytingAlgebra.toLattice α c) le eq_le sup eq_sup inf eq_inf
  __ := OrderBot.copy (@GeneralizedCoheytingAlgebra.toOrderBot α c) bot eq_bot
        /-
          α : Type u
          c : GeneralizedCoheytingAlgebra α
          le : α → α → Prop
          eq_le : Eq le LE.le
          bot : α
          eq_bot : Eq bot Bot.bot
          sup : α → α → α
          eq_sup : Eq sup Max.max
          inf : α → α → α
          eq_inf : Eq inf Min.min
          sdiff : α → α → α
          eq_sdiff : Eq sdiff SDiff.sdiff
          ⊢ ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
        -/
    (by rw [← eq_le]; exact fun _ _ ↦ .rfl)
                      /-
                        🎉 no goals
                      -/
  sdiff := sdiff
                     /-
                       α : Type u
                       c : GeneralizedCoheytingAlgebra α
                       le : α → α → Prop
                       eq_le : Eq le LE.le
                       bot : α
                       eq_bot : Eq bot Bot.bot
                       sup : α → α → α
                       eq_sup : Eq sup Max.max
                       inf : α → α → α
                       eq_inf : Eq inf Min.min
                       sdiff : α → α → α
                       eq_sdiff : Eq sdiff SDiff.sdiff
                       ⊢ ∀ (a b c : α), Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max b c))
                     -/
  sdiff_le_iff := by simp [eq_le, eq_sdiff, eq_sup]
                     /-
                       🎉 no goals
                     -/


/-- A function to create a provable equal copy of a heyting algebra
with possibly different definitional equalities. -/
def HeytingAlgebra.copy (c : HeytingAlgebra α)
                                          /-
                                            α : Type u
                                            c : HeytingAlgebra α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : HeytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : HeytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : HeytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : HeytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                             /-
                                               α : Type u
                                               c : HeytingAlgebra α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               himp : α → α → α
                                               ⊢ HImp α
                                             -/
    (himp : α → α → α) (eq_himp : himp = (by infer_instance : HImp α).himp)
                                             /-
                                               🎉 no goals
                                             -/
                                            /-
                                              α : Type u
                                              c : HeytingAlgebra α
                                              le : α → α → Prop
                                              eq_le : Eq le LE.le
                                              top : α
                                              eq_top : Eq top Top.top
                                              bot : α
                                              eq_bot : Eq bot Bot.bot
                                              sup : α → α → α
                                              eq_sup : Eq sup Max.max
                                              inf : α → α → α
                                              eq_inf : Eq inf Min.min
                                              himp : α → α → α
                                              eq_himp : Eq himp HImp.himp
                                              compl : α → α
                                              ⊢ HasCompl α
                                            -/
    (compl : α → α) (eq_compl : compl = (by infer_instance : HasCompl α).compl) :
                                            /-
                                              🎉 no goals
                                            -/
    HeytingAlgebra α where
  toGeneralizedHeytingAlgebra := GeneralizedHeytingAlgebra.copy
    (@HeytingAlgebra.toGeneralizedHeytingAlgebra α c) le eq_le top eq_top sup eq_sup inf eq_inf himp
    eq_himp
  __ := OrderBot.copy (@HeytingAlgebra.toOrderBot α c) bot eq_bot
        /-
          α : Type u
          c : HeytingAlgebra α
          le : α → α → Prop
          eq_le : Eq le LE.le
          top : α
          eq_top : Eq top Top.top
          bot : α
          eq_bot : Eq bot Bot.bot
          sup : α → α → α
          eq_sup : Eq sup Max.max
          inf : α → α → α
          eq_inf : Eq inf Min.min
          himp : α → α → α
          eq_himp : Eq himp HImp.himp
          compl : α → α
          eq_compl : Eq compl HasCompl.compl
          ⊢ ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
        -/
    (by rw [← eq_le]; exact fun _ _ ↦ .rfl)
                      /-
                        🎉 no goals
                      -/
  compl := compl
                 /-
                   α : Type u
                   c : HeytingAlgebra α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   top : α
                   eq_top : Eq top Top.top
                   bot : α
                   eq_bot : Eq bot Bot.bot
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   himp : α → α → α
                   eq_himp : Eq himp HImp.himp
                   compl : α → α
                   eq_compl : Eq compl HasCompl.compl
                   ⊢ ∀ (a : α), Eq (HImp.himp a Bot.bot) (HasCompl.compl a)
                 -/
  himp_bot := by simp [eq_le, eq_himp, eq_bot, eq_compl]
                 /-
                   🎉 no goals
                 -/


/-- A function to create a provable equal copy of a coheyting algebra
with possibly different definitional equalities. -/
def CoheytingAlgebra.copy (c : CoheytingAlgebra α)
                                          /-
                                            α : Type u
                                            c : CoheytingAlgebra α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : CoheytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : CoheytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : CoheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : CoheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                                /-
                                                  α : Type u
                                                  c : CoheytingAlgebra α
                                                  le : α → α → Prop
                                                  eq_le : Eq le LE.le
                                                  top : α
                                                  eq_top : Eq top Top.top
                                                  bot : α
                                                  eq_bot : Eq bot Bot.bot
                                                  sup : α → α → α
                                                  eq_sup : Eq sup Max.max
                                                  inf : α → α → α
                                                  eq_inf : Eq inf Min.min
                                                  sdiff : α → α → α
                                                  ⊢ SDiff α
                                                -/
    (sdiff : α → α → α) (eq_sdiff : sdiff = (by infer_instance : SDiff α).sdiff)
                                                /-
                                                  🎉 no goals
                                                -/
                                         /-
                                           α : Type u
                                           c : CoheytingAlgebra α
                                           le : α → α → Prop
                                           eq_le : Eq le LE.le
                                           top : α
                                           eq_top : Eq top Top.top
                                           bot : α
                                           eq_bot : Eq bot Bot.bot
                                           sup : α → α → α
                                           eq_sup : Eq sup Max.max
                                           inf : α → α → α
                                           eq_inf : Eq inf Min.min
                                           sdiff : α → α → α
                                           eq_sdiff : Eq sdiff SDiff.sdiff
                                           hnot : α → α
                                           ⊢ HNot α
                                         -/
    (hnot : α → α) (eq_hnot : hnot = (by infer_instance : HNot α).hnot) :
                                         /-
                                           🎉 no goals
                                         -/
    CoheytingAlgebra α where
  toGeneralizedCoheytingAlgebra := GeneralizedCoheytingAlgebra.copy
    (@CoheytingAlgebra.toGeneralizedCoheytingAlgebra α c) le eq_le bot eq_bot sup eq_sup inf eq_inf
      sdiff eq_sdiff
  __ := OrderTop.copy (@CoheytingAlgebra.toOrderTop α c) top eq_top
        /-
          α : Type u
          c : CoheytingAlgebra α
          le : α → α → Prop
          eq_le : Eq le LE.le
          top : α
          eq_top : Eq top Top.top
          bot : α
          eq_bot : Eq bot Bot.bot
          sup : α → α → α
          eq_sup : Eq sup Max.max
          inf : α → α → α
          eq_inf : Eq inf Min.min
          sdiff : α → α → α
          eq_sdiff : Eq sdiff SDiff.sdiff
          hnot : α → α
          eq_hnot : Eq hnot HNot.hnot
          ⊢ ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
        -/
    (by rw [← eq_le]; exact fun _ _ ↦ .rfl)
                      /-
                        🎉 no goals
                      -/
  hnot := hnot
                  /-
                    α : Type u
                    c : CoheytingAlgebra α
                    le : α → α → Prop
                    eq_le : Eq le LE.le
                    top : α
                    eq_top : Eq top Top.top
                    bot : α
                    eq_bot : Eq bot Bot.bot
                    sup : α → α → α
                    eq_sup : Eq sup Max.max
                    inf : α → α → α
                    eq_inf : Eq inf Min.min
                    sdiff : α → α → α
                    eq_sdiff : Eq sdiff SDiff.sdiff
                    hnot : α → α
                    eq_hnot : Eq hnot HNot.hnot
                    ⊢ ∀ (a : α), Eq (SDiff.sdiff Top.top a) (HNot.hnot a)
                  -/
  top_sdiff := by simp [eq_le, eq_sdiff, eq_top, eq_hnot]
                  /-
                    🎉 no goals
                  -/


/-- A function to create a provable equal copy of a biheyting algebra
with possibly different definitional equalities. -/
def BiheytingAlgebra.copy (c : BiheytingAlgebra α)
                                          /-
                                            α : Type u
                                            c : BiheytingAlgebra α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : BiheytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : BiheytingAlgebra α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : BiheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : BiheytingAlgebra α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                                /-
                                                  α : Type u
                                                  c : BiheytingAlgebra α
                                                  le : α → α → Prop
                                                  eq_le : Eq le LE.le
                                                  top : α
                                                  eq_top : Eq top Top.top
                                                  bot : α
                                                  eq_bot : Eq bot Bot.bot
                                                  sup : α → α → α
                                                  eq_sup : Eq sup Max.max
                                                  inf : α → α → α
                                                  eq_inf : Eq inf Min.min
                                                  sdiff : α → α → α
                                                  ⊢ SDiff α
                                                -/
    (sdiff : α → α → α) (eq_sdiff : sdiff = (by infer_instance : SDiff α).sdiff)
                                                /-
                                                  🎉 no goals
                                                -/
                                         /-
                                           α : Type u
                                           c : BiheytingAlgebra α
                                           le : α → α → Prop
                                           eq_le : Eq le LE.le
                                           top : α
                                           eq_top : Eq top Top.top
                                           bot : α
                                           eq_bot : Eq bot Bot.bot
                                           sup : α → α → α
                                           eq_sup : Eq sup Max.max
                                           inf : α → α → α
                                           eq_inf : Eq inf Min.min
                                           sdiff : α → α → α
                                           eq_sdiff : Eq sdiff SDiff.sdiff
                                           hnot : α → α
                                           ⊢ HNot α
                                         -/
    (hnot : α → α) (eq_hnot : hnot = (by infer_instance : HNot α).hnot)
                                         /-
                                           🎉 no goals
                                         -/
                                             /-
                                               α : Type u
                                               c : BiheytingAlgebra α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               himp : α → α → α
                                               ⊢ HImp α
                                             -/
    (himp : α → α → α) (eq_himp : himp = (by infer_instance : HImp α).himp)
                                             /-
                                               🎉 no goals
                                             -/
                                            /-
                                              α : Type u
                                              c : BiheytingAlgebra α
                                              le : α → α → Prop
                                              eq_le : Eq le LE.le
                                              top : α
                                              eq_top : Eq top Top.top
                                              bot : α
                                              eq_bot : Eq bot Bot.bot
                                              sup : α → α → α
                                              eq_sup : Eq sup Max.max
                                              inf : α → α → α
                                              eq_inf : Eq inf Min.min
                                              sdiff : α → α → α
                                              eq_sdiff : Eq sdiff SDiff.sdiff
                                              hnot : α → α
                                              eq_hnot : Eq hnot HNot.hnot
                                              himp : α → α → α
                                              eq_himp : Eq himp HImp.himp
                                              compl : α → α
                                              ⊢ HasCompl α
                                            -/
    (compl : α → α) (eq_compl : compl = (by infer_instance : HasCompl α).compl) :
                                            /-
                                              🎉 no goals
                                            -/
    BiheytingAlgebra α where
  toHeytingAlgebra := HeytingAlgebra.copy (@BiheytingAlgebra.toHeytingAlgebra α c) le eq_le top
    eq_top bot eq_bot sup eq_sup inf eq_inf himp eq_himp compl eq_compl
  __ := CoheytingAlgebra.copy (@BiheytingAlgebra.toCoheytingAlgebra α c) le eq_le top eq_top bot
    eq_bot sup eq_sup inf eq_inf sdiff eq_sdiff hnot eq_hnot


/-- A function to create a provable equal copy of a complete lattice
with possibly different definitional equalities. -/
def CompleteLattice.copy (c : CompleteLattice α)
                                          /-
                                            α : Type u
                                            c : CompleteLattice α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : CompleteLattice α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : CompleteLattice α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : CompleteLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : CompleteLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                             /-
                                               α : Type u
                                               c : CompleteLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sSup : Set α → α
                                               ⊢ SupSet α
                                             -/
    (sSup : Set α → α) (eq_sSup : sSup = (by infer_instance : SupSet α).sSup)
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               α : Type u
                                               c : CompleteLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sSup : Set α → α
                                               eq_sSup : Eq sSup SupSet.sSup
                                               sInf : Set α → α
                                               ⊢ InfSet α
                                             -/
    (sInf : Set α → α) (eq_sInf : sInf = (by infer_instance : InfSet α).sInf) :
                                             /-
                                               🎉 no goals
                                             -/
    CompleteLattice α where
  toLattice := Lattice.copy (@CompleteLattice.toLattice α c) le eq_le sup eq_sup inf eq_inf
  top := top
  bot := bot
  sSup := sSup
  sInf := sInf
                /-
                  α : Type u
                  c : CompleteLattice α
                  le : α → α → Prop
                  eq_le : Eq le LE.le
                  top : α
                  eq_top : Eq top Top.top
                  bot : α
                  eq_bot : Eq bot Bot.bot
                  sup : α → α → α
                  eq_sup : Eq sup Max.max
                  inf : α → α → α
                  eq_inf : Eq inf Min.min
                  sSup : Set α → α
                  eq_sSup : Eq sSup SupSet.sSup
                  sInf : Set α → α
                  eq_sInf : Eq sInf InfSet.sInf
                  ⊢ ∀ (s : Set α) (a : α), Membership.mem s a → LE.le a (SupSet.sSup s)
                -/
  le_sSup := by intro _ _ h; simp [eq_le, eq_sSup, le_sSup _ _ h]
                             /-
                               🎉 no goals
                             -/
                /-
                  α : Type u
                  c : CompleteLattice α
                  le : α → α → Prop
                  eq_le : Eq le LE.le
                  top : α
                  eq_top : Eq top Top.top
                  bot : α
                  eq_bot : Eq bot Bot.bot
                  sup : α → α → α
                  eq_sup : Eq sup Max.max
                  inf : α → α → α
                  eq_inf : Eq inf Min.min
                  sSup : Set α → α
                  eq_sSup : Eq sSup SupSet.sSup
                  sInf : Set α → α
                  eq_sInf : Eq sInf InfSet.sInf
                  ⊢ ∀ (s : Set α) (a : α), (∀ (b : α), Membership.mem s b → LE.le b a) → LE.le ( …
                -/
  sSup_le := by intro _ _ h; simpa [eq_le, eq_sSup] using h
                             /-
                               🎉 no goals
                             -/
                /-
                  α : Type u
                  c : CompleteLattice α
                  le : α → α → Prop
                  eq_le : Eq le LE.le
                  top : α
                  eq_top : Eq top Top.top
                  bot : α
                  eq_bot : Eq bot Bot.bot
                  sup : α → α → α
                  eq_sup : Eq sup Max.max
                  inf : α → α → α
                  eq_inf : Eq inf Min.min
                  sSup : Set α → α
                  eq_sSup : Eq sSup SupSet.sSup
                  sInf : Set α → α
                  eq_sInf : Eq sInf InfSet.sInf
                  ⊢ ∀ (s : Set α) (a : α), Membership.mem s a → LE.le (InfSet.sInf s) a
                -/
  sInf_le := by intro _ _ h; simp [eq_le, eq_sInf, sInf_le _ _ h]
                             /-
                               🎉 no goals
                             -/
                /-
                  α : Type u
                  c : CompleteLattice α
                  le : α → α → Prop
                  eq_le : Eq le LE.le
                  top : α
                  eq_top : Eq top Top.top
                  bot : α
                  eq_bot : Eq bot Bot.bot
                  sup : α → α → α
                  eq_sup : Eq sup Max.max
                  inf : α → α → α
                  eq_inf : Eq inf Min.min
                  sSup : Set α → α
                  eq_sSup : Eq sSup SupSet.sSup
                  sInf : Set α → α
                  eq_sInf : Eq sInf InfSet.sInf
                  ⊢ ∀ (s : Set α) (a : α), (∀ (b : α), Membership.mem s b → LE.le a b) → LE.le a …
                -/
  le_sInf := by intro _ _ h; simpa [eq_le, eq_sInf] using h
                             /-
                               🎉 no goals
                             -/
               /-
                 α : Type u
                 c : CompleteLattice α
                 le : α → α → Prop
                 eq_le : Eq le LE.le
                 top : α
                 eq_top : Eq top Top.top
                 bot : α
                 eq_bot : Eq bot Bot.bot
                 sup : α → α → α
                 eq_sup : Eq sup Max.max
                 inf : α → α → α
                 eq_inf : Eq inf Min.min
                 sSup : Set α → α
                 eq_sSup : Eq sSup SupSet.sSup
                 sInf : Set α → α
                 eq_sInf : Eq sInf InfSet.sInf
                 ⊢ ∀ (x : α), LE.le x Top.top
               -/
  le_top := by intros; simp [eq_le, eq_top]
                       /-
                         🎉 no goals
                       -/
               /-
                 α : Type u
                 c : CompleteLattice α
                 le : α → α → Prop
                 eq_le : Eq le LE.le
                 top : α
                 eq_top : Eq top Top.top
                 bot : α
                 eq_bot : Eq bot Bot.bot
                 sup : α → α → α
                 eq_sup : Eq sup Max.max
                 inf : α → α → α
                 eq_inf : Eq inf Min.min
                 sSup : Set α → α
                 eq_sSup : Eq sSup SupSet.sSup
                 sInf : Set α → α
                 eq_sInf : Eq sInf InfSet.sInf
                 ⊢ ∀ (x : α), LE.le Bot.bot x
               -/
  bot_le := by intros; simp [eq_le, eq_bot]
                       /-
                         🎉 no goals
                       -/


/-- A function to create a provable equal copy of a frame with possibly different definitional
equalities. -/
                                                                   /-
                                                                     α : Type u
                                                                     c : Order.Frame α
                                                                     le : α → α → Prop
                                                                     ⊢ LE α
                                                                   -/
def Frame.copy (c : Frame α) (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                  /-
                                    α : Type u
                                    c : Order.Frame α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : Order.Frame α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : Order.Frame α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : Order.Frame α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                             /-
                                               α : Type u
                                               c : Order.Frame α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               himp : α → α → α
                                               ⊢ HImp α
                                             -/
    (himp : α → α → α) (eq_himp : himp = (by infer_instance : HImp α).himp)
                                             /-
                                               🎉 no goals
                                             -/
                                            /-
                                              α : Type u
                                              c : Order.Frame α
                                              le : α → α → Prop
                                              eq_le : Eq le LE.le
                                              top : α
                                              eq_top : Eq top Top.top
                                              bot : α
                                              eq_bot : Eq bot Bot.bot
                                              sup : α → α → α
                                              eq_sup : Eq sup Max.max
                                              inf : α → α → α
                                              eq_inf : Eq inf Min.min
                                              himp : α → α → α
                                              eq_himp : Eq himp HImp.himp
                                              compl : α → α
                                              ⊢ HasCompl α
                                            -/
    (compl : α → α) (eq_compl : compl = (by infer_instance : HasCompl α).compl)
                                            /-
                                              🎉 no goals
                                            -/
                                             /-
                                               α : Type u
                                               c : Order.Frame α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               himp : α → α → α
                                               eq_himp : Eq himp HImp.himp
                                               compl : α → α
                                               eq_compl : Eq compl HasCompl.compl
                                               sSup : Set α → α
                                               ⊢ SupSet α
                                             -/
    (sSup : Set α → α) (eq_sSup : sSup = (by infer_instance : SupSet α).sSup)
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               α : Type u
                                               c : Order.Frame α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               himp : α → α → α
                                               eq_himp : Eq himp HImp.himp
                                               compl : α → α
                                               eq_compl : Eq compl HasCompl.compl
                                               sSup : Set α → α
                                               eq_sSup : Eq sSup SupSet.sSup
                                               sInf : Set α → α
                                               ⊢ InfSet α
                                             -/
    (sInf : Set α → α) (eq_sInf : sInf = (by infer_instance : InfSet α).sInf) : Frame α where
                                             /-
                                               🎉 no goals
                                             -/
  toCompleteLattice := CompleteLattice.copy (@Frame.toCompleteLattice α c)
    le eq_le top eq_top bot eq_bot sup eq_sup inf eq_inf sSup eq_sSup sInf eq_sInf
  inf_sSup_le_iSup_inf := fun a s => by
    /-
      α : Type u
      c : Order.Frame α
      le : α → α → Prop
      eq_le : Eq le LE.le
      top : α
      eq_top : Eq top Top.top
      bot : α
      eq_bot : Eq bot Bot.bot
      sup : α → α → α
      eq_sup : Eq sup Max.max
      inf : α → α → α
      eq_inf : Eq inf Min.min
      himp : α → α → α
      eq_himp : Eq himp HImp.himp
      compl : α → α
      eq_compl : Eq compl HasCompl.compl
      sSup : Set α → α
      eq_sSup : Eq sSup SupSet.sSup
      sInf : Set α → α
      eq_sInf : Eq sInf InfSet.sInf
      a : α
      s : Set α
      ⊢ LE.le (Min.min a (SupSet.sSup s)) (iSup fun b => iSup fun h => Min.min a b)
    -/
    simp [eq_le, eq_sup, eq_inf, eq_sSup, @Order.Frame.inf_sSup_le_iSup_inf α _ a s]
    /-
      🎉 no goals
    -/
  __ := HeytingAlgebra.copy (@Frame.toHeytingAlgebra α c)
    le eq_le top eq_top bot eq_bot sup eq_sup inf eq_inf himp eq_himp compl eq_compl


/-- A function to create a provable equal copy of a coframe with possibly different definitional
equalities. -/
                                                                       /-
                                                                         α : Type u
                                                                         c : Order.Coframe α
                                                                         le : α → α → Prop
                                                                         ⊢ LE α
                                                                       -/
def Coframe.copy (c : Coframe α) (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                  /-
                                    α : Type u
                                    c : Order.Coframe α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : Order.Coframe α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : Order.Coframe α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : Order.Coframe α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                                /-
                                                  α : Type u
                                                  c : Order.Coframe α
                                                  le : α → α → Prop
                                                  eq_le : Eq le LE.le
                                                  top : α
                                                  eq_top : Eq top Top.top
                                                  bot : α
                                                  eq_bot : Eq bot Bot.bot
                                                  sup : α → α → α
                                                  eq_sup : Eq sup Max.max
                                                  inf : α → α → α
                                                  eq_inf : Eq inf Min.min
                                                  sdiff : α → α → α
                                                  ⊢ SDiff α
                                                -/
    (sdiff : α → α → α) (eq_sdiff : sdiff = (by infer_instance : SDiff α).sdiff)
                                                /-
                                                  🎉 no goals
                                                -/
                                         /-
                                           α : Type u
                                           c : Order.Coframe α
                                           le : α → α → Prop
                                           eq_le : Eq le LE.le
                                           top : α
                                           eq_top : Eq top Top.top
                                           bot : α
                                           eq_bot : Eq bot Bot.bot
                                           sup : α → α → α
                                           eq_sup : Eq sup Max.max
                                           inf : α → α → α
                                           eq_inf : Eq inf Min.min
                                           sdiff : α → α → α
                                           eq_sdiff : Eq sdiff SDiff.sdiff
                                           hnot : α → α
                                           ⊢ HNot α
                                         -/
    (hnot : α → α) (eq_hnot : hnot = (by infer_instance : HNot α).hnot)
                                         /-
                                           🎉 no goals
                                         -/
                                             /-
                                               α : Type u
                                               c : Order.Coframe α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               sSup : Set α → α
                                               ⊢ SupSet α
                                             -/
    (sSup : Set α → α) (eq_sSup : sSup = (by infer_instance : SupSet α).sSup)
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               α : Type u
                                               c : Order.Coframe α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               sSup : Set α → α
                                               eq_sSup : Eq sSup SupSet.sSup
                                               sInf : Set α → α
                                               ⊢ InfSet α
                                             -/
    (sInf : Set α → α) (eq_sInf : sInf = (by infer_instance : InfSet α).sInf) : Coframe α where
                                             /-
                                               🎉 no goals
                                             -/
  toCompleteLattice := CompleteLattice.copy (@Coframe.toCompleteLattice α c)
    le eq_le top eq_top bot eq_bot sup eq_sup inf eq_inf sSup eq_sSup sInf eq_sInf
  iInf_sup_le_sup_sInf := fun a s => by
    /-
      α : Type u
      c : Order.Coframe α
      le : α → α → Prop
      eq_le : Eq le LE.le
      top : α
      eq_top : Eq top Top.top
      bot : α
      eq_bot : Eq bot Bot.bot
      sup : α → α → α
      eq_sup : Eq sup Max.max
      inf : α → α → α
      eq_inf : Eq inf Min.min
      sdiff : α → α → α
      eq_sdiff : Eq sdiff SDiff.sdiff
      hnot : α → α
      eq_hnot : Eq hnot HNot.hnot
      sSup : Set α → α
      eq_sSup : Eq sSup SupSet.sSup
      sInf : Set α → α
      eq_sInf : Eq sInf InfSet.sInf
      a : α
      s : Set α
      ⊢ LE.le (iInf fun b => iInf fun h => Max.max a b) (Max.max a (InfSet.sInf s))
    -/
    simp [eq_le, eq_sup, eq_inf, eq_sInf, @Order.Coframe.iInf_sup_le_sup_sInf α _ a s]
    /-
      🎉 no goals
    -/
  __ := CoheytingAlgebra.copy (@Coframe.toCoheytingAlgebra α c)
    le eq_le top eq_top bot eq_bot sup eq_sup inf eq_inf sdiff eq_sdiff hnot eq_hnot


/-- A function to create a provable equal copy of a complete distributive lattice
with possibly different definitional equalities. -/
def CompleteDistribLattice.copy (c : CompleteDistribLattice α)
                                          /-
                                            α : Type u
                                            c : CompleteDistribLattice α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    α : Type u
                                    c : CompleteDistribLattice α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    ⊢ Top α
                                  -/
    (top : α) (eq_top : top = (by infer_instance : Top α).top)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    c : CompleteDistribLattice α
                                    le : α → α → Prop
                                    eq_le : Eq le LE.le
                                    top : α
                                    eq_top : Eq top Top.top
                                    bot : α
                                    ⊢ Bot α
                                  -/
    (bot : α) (eq_bot : bot = (by infer_instance : Bot α).bot)
                                  /-
                                    🎉 no goals
                                  -/
                                          /-
                                            α : Type u
                                            c : CompleteDistribLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : CompleteDistribLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            top : α
                                            eq_top : Eq top Top.top
                                            bot : α
                                            eq_bot : Eq bot Bot.bot
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                                /-
                                                  α : Type u
                                                  c : CompleteDistribLattice α
                                                  le : α → α → Prop
                                                  eq_le : Eq le LE.le
                                                  top : α
                                                  eq_top : Eq top Top.top
                                                  bot : α
                                                  eq_bot : Eq bot Bot.bot
                                                  sup : α → α → α
                                                  eq_sup : Eq sup Max.max
                                                  inf : α → α → α
                                                  eq_inf : Eq inf Min.min
                                                  sdiff : α → α → α
                                                  ⊢ SDiff α
                                                -/
    (sdiff : α → α → α) (eq_sdiff : sdiff = (by infer_instance : SDiff α).sdiff)
                                                /-
                                                  🎉 no goals
                                                -/
                                         /-
                                           α : Type u
                                           c : CompleteDistribLattice α
                                           le : α → α → Prop
                                           eq_le : Eq le LE.le
                                           top : α
                                           eq_top : Eq top Top.top
                                           bot : α
                                           eq_bot : Eq bot Bot.bot
                                           sup : α → α → α
                                           eq_sup : Eq sup Max.max
                                           inf : α → α → α
                                           eq_inf : Eq inf Min.min
                                           sdiff : α → α → α
                                           eq_sdiff : Eq sdiff SDiff.sdiff
                                           hnot : α → α
                                           ⊢ HNot α
                                         -/
    (hnot : α → α) (eq_hnot : hnot = (by infer_instance : HNot α).hnot)
                                         /-
                                           🎉 no goals
                                         -/
                                             /-
                                               α : Type u
                                               c : CompleteDistribLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               himp : α → α → α
                                               ⊢ HImp α
                                             -/
    (himp : α → α → α) (eq_himp : himp = (by infer_instance : HImp α).himp)
                                             /-
                                               🎉 no goals
                                             -/
                                            /-
                                              α : Type u
                                              c : CompleteDistribLattice α
                                              le : α → α → Prop
                                              eq_le : Eq le LE.le
                                              top : α
                                              eq_top : Eq top Top.top
                                              bot : α
                                              eq_bot : Eq bot Bot.bot
                                              sup : α → α → α
                                              eq_sup : Eq sup Max.max
                                              inf : α → α → α
                                              eq_inf : Eq inf Min.min
                                              sdiff : α → α → α
                                              eq_sdiff : Eq sdiff SDiff.sdiff
                                              hnot : α → α
                                              eq_hnot : Eq hnot HNot.hnot
                                              himp : α → α → α
                                              eq_himp : Eq himp HImp.himp
                                              compl : α → α
                                              ⊢ HasCompl α
                                            -/
    (compl : α → α) (eq_compl : compl = (by infer_instance : HasCompl α).compl)
                                            /-
                                              🎉 no goals
                                            -/
                                             /-
                                               α : Type u
                                               c : CompleteDistribLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               himp : α → α → α
                                               eq_himp : Eq himp HImp.himp
                                               compl : α → α
                                               eq_compl : Eq compl HasCompl.compl
                                               sSup : Set α → α
                                               ⊢ SupSet α
                                             -/
    (sSup : Set α → α) (eq_sSup : sSup = (by infer_instance : SupSet α).sSup)
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               α : Type u
                                               c : CompleteDistribLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               top : α
                                               eq_top : Eq top Top.top
                                               bot : α
                                               eq_bot : Eq bot Bot.bot
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sdiff : α → α → α
                                               eq_sdiff : Eq sdiff SDiff.sdiff
                                               hnot : α → α
                                               eq_hnot : Eq hnot HNot.hnot
                                               himp : α → α → α
                                               eq_himp : Eq himp HImp.himp
                                               compl : α → α
                                               eq_compl : Eq compl HasCompl.compl
                                               sSup : Set α → α
                                               eq_sSup : Eq sSup SupSet.sSup
                                               sInf : Set α → α
                                               ⊢ InfSet α
                                             -/
    (sInf : Set α → α) (eq_sInf : sInf = (by infer_instance : InfSet α).sInf) :
                                             /-
                                               🎉 no goals
                                             -/
    CompleteDistribLattice α where
  toFrame := Frame.copy (@CompleteDistribLattice.toFrame α c) le eq_le top eq_top bot eq_bot sup
    eq_sup inf eq_inf himp eq_himp compl eq_compl sSup eq_sSup sInf eq_sInf
  __ := Coframe.copy (@CompleteDistribLattice.toCoframe α c) le eq_le top eq_top bot eq_bot sup
    eq_sup inf eq_inf sdiff eq_sdiff hnot eq_hnot sSup eq_sSup sInf eq_sInf


/-- A function to create a provable equal copy of a conditionally complete lattice
with possibly different definitional equalities. -/
def ConditionallyCompleteLattice.copy (c : ConditionallyCompleteLattice α)
                                          /-
                                            α : Type u
                                            c : ConditionallyCompleteLattice α
                                            le : α → α → Prop
                                            ⊢ LE α
                                          -/
    (le : α → α → Prop) (eq_le : le = (by infer_instance : LE α).le)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : ConditionallyCompleteLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            ⊢ Max α
                                          -/
    (sup : α → α → α) (eq_sup : sup = (by infer_instance : Max α).max)
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            c : ConditionallyCompleteLattice α
                                            le : α → α → Prop
                                            eq_le : Eq le LE.le
                                            sup : α → α → α
                                            eq_sup : Eq sup Max.max
                                            inf : α → α → α
                                            ⊢ Min α
                                          -/
    (inf : α → α → α) (eq_inf : inf = (by infer_instance : Min α).min)
                                          /-
                                            🎉 no goals
                                          -/
                                             /-
                                               α : Type u
                                               c : ConditionallyCompleteLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sSup : Set α → α
                                               ⊢ SupSet α
                                             -/
    (sSup : Set α → α) (eq_sSup : sSup = (by infer_instance : SupSet α).sSup)
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               α : Type u
                                               c : ConditionallyCompleteLattice α
                                               le : α → α → Prop
                                               eq_le : Eq le LE.le
                                               sup : α → α → α
                                               eq_sup : Eq sup Max.max
                                               inf : α → α → α
                                               eq_inf : Eq inf Min.min
                                               sSup : Set α → α
                                               eq_sSup : Eq sSup SupSet.sSup
                                               sInf : Set α → α
                                               ⊢ InfSet α
                                             -/
    (sInf : Set α → α) (eq_sInf : sInf = (by infer_instance : InfSet α).sInf) :
                                             /-
                                               🎉 no goals
                                             -/
    ConditionallyCompleteLattice α where
  toLattice := Lattice.copy (@ConditionallyCompleteLattice.toLattice α c)
    le eq_le sup eq_sup inf eq_inf
  sSup := sSup
  sInf := sInf
                 /-
                   α : Type u
                   c : ConditionallyCompleteLattice α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   sSup : Set α → α
                   eq_sSup : Eq sSup SupSet.sSup
                   sInf : Set α → α
                   eq_sInf : Eq sInf InfSet.sInf
                   ⊢ ∀ (s : Set α) (a : α), BddAbove s → Membership.mem s a → LE.le a (SupSet.sSu …
                 -/
  le_csSup := by intro _ _ hb h; subst_vars; exact le_csSup _ _ hb h
                                             /-
                                               🎉 no goals
                                             -/
                 /-
                   α : Type u
                   c : ConditionallyCompleteLattice α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   sSup : Set α → α
                   eq_sSup : Eq sSup SupSet.sSup
                   sInf : Set α → α
                   eq_sInf : Eq sInf InfSet.sInf
                   ⊢ ∀ (s : Set α) (a : α), s.Nonempty → Membership.mem (upperBounds s) a → LE.le …
                 -/
  csSup_le := by intro _ _ hb h; subst_vars; exact csSup_le _ _ hb h
                                             /-
                                               🎉 no goals
                                             -/
                 /-
                   α : Type u
                   c : ConditionallyCompleteLattice α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   sSup : Set α → α
                   eq_sSup : Eq sSup SupSet.sSup
                   sInf : Set α → α
                   eq_sInf : Eq sInf InfSet.sInf
                   ⊢ ∀ (s : Set α) (a : α), BddBelow s → Membership.mem s a → LE.le (InfSet.sInf  …
                 -/
  csInf_le := by intro _ _ hb h; subst_vars; exact csInf_le _ _ hb h
                                             /-
                                               🎉 no goals
                                             -/
                 /-
                   α : Type u
                   c : ConditionallyCompleteLattice α
                   le : α → α → Prop
                   eq_le : Eq le LE.le
                   sup : α → α → α
                   eq_sup : Eq sup Max.max
                   inf : α → α → α
                   eq_inf : Eq inf Min.min
                   sSup : Set α → α
                   eq_sSup : Eq sSup SupSet.sSup
                   sInf : Set α → α
                   eq_sInf : Eq sInf InfSet.sInf
                   ⊢ ∀ (s : Set α) (a : α), s.Nonempty → Membership.mem (lowerBounds s) a → LE.le …
                 -/
  le_csInf := by intro _ _ hb h; subst_vars; exact le_csInf _ _ hb h
                                             /-
                                               🎉 no goals
                                             -/

