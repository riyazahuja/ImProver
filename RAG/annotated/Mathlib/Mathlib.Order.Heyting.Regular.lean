/-- An element of a Heyting algebra is regular if its double complement is itself. -/
def IsRegular (a : α) : Prop :=
  aᶜᶜ = a


protected theorem IsRegular.eq : IsRegular a → aᶜᶜ = a :=
  id


instance IsRegular.decidablePred [DecidableEq α] : @DecidablePred α IsRegular := fun _ =>
  ‹DecidableEq α› _ _


                                                /-
                                                  α : Type u_1
                                                  inst✝ : HeytingAlgebra α
                                                  ⊢ Heyting.IsRegular Bot.bot
                                                -/
theorem isRegular_bot : IsRegular (⊥ : α) := by rw [IsRegular, compl_bot, compl_top]
                                                /-
                                                  🎉 no goals
                                                -/


                                                /-
                                                  α : Type u_1
                                                  inst✝ : HeytingAlgebra α
                                                  ⊢ Heyting.IsRegular Top.top
                                                -/
theorem isRegular_top : IsRegular (⊤ : α) := by rw [IsRegular, compl_top, compl_bot]
                                                /-
                                                  🎉 no goals
                                                -/


theorem IsRegular.inf (ha : IsRegular a) (hb : IsRegular b) : IsRegular (a ⊓ b) := by
  /-
    α : Type u_1
    inst✝ : HeytingAlgebra α
    a b : α
    ha : Heyting.IsRegular a
    hb : Heyting.IsRegular b
    ⊢ Heyting.IsRegular (Min.min a b)
  -/
  rw [IsRegular, compl_compl_inf_distrib, ha.eq, hb.eq]
  /-
    🎉 no goals
  -/


theorem IsRegular.himp (ha : IsRegular a) (hb : IsRegular b) : IsRegular (a ⇨ b) := by
  /-
    α : Type u_1
    inst✝ : HeytingAlgebra α
    a b : α
    ha : Heyting.IsRegular a
    hb : Heyting.IsRegular b
    ⊢ Heyting.IsRegular (HImp.himp a b)
  -/
  rw [IsRegular, compl_compl_himp_distrib, ha.eq, hb.eq]
  /-
    🎉 no goals
  -/


theorem isRegular_compl (a : α) : IsRegular aᶜ :=
  compl_compl_compl _


protected theorem IsRegular.disjoint_compl_left_iff (ha : IsRegular a) :
                                /-
                                  α : Type u_1
                                  inst✝ : HeytingAlgebra α
                                  a b : α
                                  ha : Heyting.IsRegular a
                                  ⊢ Iff (Disjoint (HasCompl.compl a) b) (LE.le b a)
                                -/
    Disjoint aᶜ b ↔ b ≤ a := by rw [← le_compl_iff_disjoint_left, ha.eq]
                                /-
                                  🎉 no goals
                                -/


protected theorem IsRegular.disjoint_compl_right_iff (hb : IsRegular b) :
                                /-
                                  α : Type u_1
                                  inst✝ : HeytingAlgebra α
                                  a b : α
                                  hb : Heyting.IsRegular b
                                  ⊢ Iff (Disjoint a (HasCompl.compl b)) (LE.le a b)
                                -/
    Disjoint a bᶜ ↔ a ≤ b := by rw [← le_compl_iff_disjoint_right, hb.eq]
                                /-
                                  🎉 no goals
                                -/

-- See note [reducible non-instances]

/-- A Heyting algebra with regular excluded middle is a boolean algebra. -/
abbrev _root_.BooleanAlgebra.ofRegular (h : ∀ a : α, IsRegular (a ⊔ aᶜ)) : BooleanAlgebra α :=
  have : ∀ a : α, IsCompl a aᶜ := fun a =>
    ⟨disjoint_compl_right,
                             /-
                               α : Type u_1
                               inst✝ : HeytingAlgebra α
                               a✝ b : α
                               h : ∀ (a : α), Heyting.IsRegular (Max.max a (HasCompl.compl a))
                               a : α
                               ⊢ Eq (Max.max a (HasCompl.compl a)) Top.top
                             -/
      codisjoint_iff.2 <| by rw [← (h a), compl_sup, inf_compl_eq_bot, compl_bot]⟩
                             /-
                               🎉 no goals
                             -/
  { ‹HeytingAlgebra α›,
    GeneralizedHeytingAlgebra.toDistribLattice with
    himp_eq := fun _ _ =>
      eq_of_forall_le_iff fun _ => le_himp_iff.trans (this _).le_sup_right_iff_inf_left_le.symm
    inf_compl_le_bot := fun _ => (this _).1.le_bot
    top_le_sup_compl := fun _ => (this _).2.top_le }


/-- The boolean algebra of Heyting regular elements. -/
def Regular : Type _ :=
  { a : α // IsRegular a }


/-- The coercion `Regular α → α` -/
@[coe] def val : Regular α → α :=
  Subtype.val


theorem prop : ∀ a : Regular α, IsRegular a.val := Subtype.prop


instance : CoeOut (Regular α) α := ⟨Regular.val⟩


theorem coe_injective : Injective ((↑) : Regular α → α) :=
  Subtype.coe_injective


@[simp]
theorem coe_inj {a b : Regular α} : (a : α) = b ↔ a = b :=
  Subtype.coe_inj


instance top : Top (Regular α) :=
  ⟨⟨⊤, isRegular_top⟩⟩


instance bot : Bot (Regular α) :=
  ⟨⟨⊥, isRegular_bot⟩⟩


instance inf : Min (Regular α) :=
  ⟨fun a b => ⟨a ⊓ b, a.2.inf b.2⟩⟩


instance himp : HImp (Regular α) :=
  ⟨fun a b => ⟨a ⇨ b, a.2.himp b.2⟩⟩


instance hasCompl : HasCompl (Regular α) :=
  ⟨fun a => ⟨aᶜ, isRegular_compl _⟩⟩


@[simp, norm_cast]
theorem coe_top : ((⊤ : Regular α) : α) = ⊤ :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : Regular α) : α) = ⊥ :=
  rfl


@[simp, norm_cast]
theorem coe_inf (a b : Regular α) : (↑(a ⊓ b) : α) = (a : α) ⊓ b :=
  rfl


@[simp, norm_cast]
theorem coe_himp (a b : Regular α) : (↑(a ⇨ b) : α) = (a : α) ⇨ b :=
  rfl


@[simp, norm_cast]
theorem coe_compl (a : Regular α) : (↑aᶜ : α) = (a : α)ᶜ :=
  rfl


instance : Inhabited (Regular α) :=
  ⟨⊥⟩


instance : SemilatticeInf (Regular α) :=
  coe_injective.semilatticeInf _ coe_inf


instance boundedOrder : BoundedOrder (Regular α) :=
  BoundedOrder.lift ((↑) : Regular α → α) (fun _ _ => id) coe_top coe_bot


@[simp, norm_cast]
theorem coe_le_coe {a b : Regular α} : (a : α) ≤ b ↔ a ≤ b :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_lt_coe {a b : Regular α} : (a : α) < b ↔ a < b :=
  Iff.rfl


/-- **Regularization** of `a`. The smallest regular element greater than `a`. -/
def toRegular : α →o Regular α :=
  ⟨fun a => ⟨aᶜᶜ, isRegular_compl _⟩, fun _ _ h =>
    coe_le_coe.1 <| compl_le_compl <| compl_le_compl h⟩


@[simp, norm_cast]
theorem coe_toRegular (a : α) : (toRegular a : α) = aᶜᶜ :=
  rfl


@[simp]
theorem toRegular_coe (a : Regular α) : toRegular (a : α) = a :=
  coe_injective a.2


/-- The Galois insertion between `Regular.toRegular` and `coe`. -/
def gi : GaloisInsertion toRegular ((↑) : Regular α → α) where
  choice a ha := ⟨a, ha.antisymm le_compl_compl⟩
  gc _ b :=
    coe_le_coe.symm.trans <|
      ⟨le_compl_compl.trans, fun h => (compl_anti <| compl_anti h).trans_eq b.2⟩
  le_l_u _ := le_compl_compl
  choice_eq _ ha := coe_injective <| le_compl_compl.antisymm ha


instance lattice : Lattice (Regular α) :=
  gi.liftLattice


@[simp, norm_cast]
theorem coe_sup (a b : Regular α) : (↑(a ⊔ b) : α) = ((a : α) ⊔ b)ᶜᶜ :=
  rfl


instance : BooleanAlgebra (Regular α) :=
  { Regular.lattice, Regular.boundedOrder, Regular.himp,
    Regular.hasCompl with
    le_sup_inf := fun a b c =>
      coe_le_coe.1 <| by
        /-
          α : Type u_1
          inst✝ : HeytingAlgebra α
          a✝ b✝ : α
          a b c : Heyting.Regular α
          ⊢ LE.le ↑(Min.min (Max.max a b) (Max.max a c)) ↑(Max.max a (Min.min b c))
        -/
        dsimp
        /-
          α : Type u_1
          inst✝ : HeytingAlgebra α
          a✝ b✝ : α
          a b c : Heyting.Regular α
          ⊢ LE.le (Min.min (HasCompl.compl (HasCompl.compl (Max.max ↑a ↑b))) (HasCompl.c …
        -/
        rw [sup_inf_left, compl_compl_inf_distrib]
        /-
          🎉 no goals
        -/
    inf_compl_le_bot := fun _ => coe_le_coe.1 <| disjoint_iff_inf_le.1 disjoint_compl_right
    top_le_sup_compl := fun a =>
      coe_le_coe.1 <| by
        /-
          α : Type u_1
          inst✝ : HeytingAlgebra α
          a✝ b : α
          a : Heyting.Regular α
          ⊢ LE.le ↑Top.top ↑(Max.max a (HasCompl.compl a))
        -/
        dsimp
        /-
          α : Type u_1
          inst✝ : HeytingAlgebra α
          a✝ b : α
          a : Heyting.Regular α
          ⊢ LE.le Top.top (HasCompl.compl (HasCompl.compl (Max.max (↑a) (HasCompl.compl  …
        -/
        rw [compl_sup, inf_compl_eq_bot, compl_bot]
        /-
          🎉 no goals
        -/
    himp_eq := fun a b =>
      coe_injective
        (by
          /-
            α : Type u_1
            inst✝ : HeytingAlgebra α
            a✝ b✝ : α
            a b : Heyting.Regular α
            ⊢ Eq ↑(HImp.himp a b) ↑(Max.max b (HasCompl.compl a))
          -/
          dsimp
          /-
            α : Type u_1
            inst✝ : HeytingAlgebra α
            a✝ b✝ : α
            a b : Heyting.Regular α
            ⊢ Eq (HImp.himp ↑a ↑b) (HasCompl.compl (HasCompl.compl (Max.max (↑b) (HasCompl …
          -/
          rw [compl_sup, a.prop.eq]
          /-
            α : Type u_1
            inst✝ : HeytingAlgebra α
            a✝ b✝ : α
            a b : Heyting.Regular α
            ⊢ Eq (HImp.himp ↑a ↑b) (HasCompl.compl (Min.min (HasCompl.compl ↑b) ↑a))
          -/
          refine eq_of_forall_le_iff fun c => le_himp_iff.trans ?_
          /-
            α : Type u_1
            inst✝ : HeytingAlgebra α
            a✝ b✝ : α
            a b : Heyting.Regular α
            c : α
            ⊢ Iff (LE.le (Min.min c ↑a) ↑b) (LE.le c (HasCompl.compl (Min.min (HasCompl.co …
          -/
          rw [le_compl_iff_disjoint_right, disjoint_left_comm]
          /-
            α : Type u_1
            inst✝ : HeytingAlgebra α
            a✝ b✝ : α
            a b : Heyting.Regular α
            c : α
            ⊢ Iff (LE.le (Min.min c ↑a) ↑b) (Disjoint (HasCompl.compl ↑b) (Min.min c ↑a))
          -/
          rw [b.prop.disjoint_compl_left_iff]) }
          /-
            🎉 no goals
          -/


@[simp, norm_cast]
theorem coe_sdiff (a b : Regular α) : (↑(a \ b) : α) = (a : α) ⊓ bᶜ :=
  rfl


theorem isRegular_of_boolean : ∀ a : α, IsRegular a :=
  compl_compl


/-- A decidable proposition is intuitionistically Heyting-regular. -/
-- Porting note: removed @[nolint decidable_classical]
theorem isRegular_of_decidable (p : Prop) [Decidable p] : IsRegular p :=
  propext <| Decidable.not_not


