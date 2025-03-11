/-- The free abelian group on a type. -/
def FreeAbelianGroup : Type u :=
  Additive <| Abelianization <| FreeGroup α

-- FIXME: this is super broken, because the functions have type `Additive .. → ..`
-- instead of `FreeAbelianGroup α → ..` and those are not defeq!

instance FreeAbelianGroup.addCommGroup : AddCommGroup (FreeAbelianGroup α) :=
  @Additive.addCommGroup _ <| Abelianization.commGroup _


instance : Inhabited (FreeAbelianGroup α) :=
  ⟨0⟩


                                                         /-
                                                           α : Type u
                                                           inst✝ : IsEmpty α
                                                           ⊢ Unique (FreeAbelianGroup α)
                                                         -/
instance [IsEmpty α] : Unique (FreeAbelianGroup α) := by unfold FreeAbelianGroup; infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The canonical map from `α` to `FreeAbelianGroup α`. -/
def of (x : α) : FreeAbelianGroup α :=
  Additive.ofMul <| Abelianization.of <| FreeGroup.of x


/-- The map `FreeAbelianGroup α →+ A` induced by a map of types `α → A`. -/
def lift {β : Type v} [AddCommGroup β] : (α → β) ≃ (FreeAbelianGroup α →+ β) :=
  (@FreeGroup.lift _ (Multiplicative β) _).trans <|
    (@Abelianization.lift _ _ (Multiplicative β) _).trans MonoidHom.toAdditive


@[simp]
protected theorem of (x : α) : lift f (of x) = f x := by
  convert Abelianization.lift.of
     (FreeGroup.lift f (β := Multiplicative β)) (FreeGroup.of x) using 1
  /-
    case h.e'_3.h
    α : Type u
    β : Type v
    inst✝ : AddCommGroup β
    f : α → β
    x : α
    e_1✝ : Eq β (Multiplicative β)
    ⊢ Eq (f x) ((FreeGroup.lift f) (FreeGroup.of x))
  -/
  exact (FreeGroup.lift.of (β := Multiplicative β)).symm
  /-
    🎉 no goals
  -/


protected theorem unique (g : FreeAbelianGroup α →+ β) (hg : ∀ x, g (of x) = f x) {x} :
    g x = lift f x :=
  DFunLike.congr_fun (lift.symm_apply_eq.mp (funext hg : g ∘ of = f)) _


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
protected theorem ext (g h : FreeAbelianGroup α →+ β) (H : ∀ x, g (of x) = h (of x)) : g = h :=
  lift.symm.injective <| funext H


theorem map_hom {α β γ} [AddCommGroup β] [AddCommGroup γ] (a : FreeAbelianGroup α) (f : α → β)
    (g : β →+ γ) : g (lift f a) = lift (g ∘ f) a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : AddCommGroup β
    inst✝ : AddCommGroup γ
    a : FreeAbelianGroup α
    f : α → β
    g : AddMonoidHom β γ
    ⊢ Eq (g ((FreeAbelianGroup.lift f) a)) ((FreeAbelianGroup.lift (Function.comp  …
  -/
  show (g.comp (lift f)) a = lift (g ∘ f) a
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : AddCommGroup β
    inst✝ : AddCommGroup γ
    a : FreeAbelianGroup α
    f : α → β
    g : AddMonoidHom β γ
    ⊢ Eq ((g.comp (FreeAbelianGroup.lift f)) a) ((FreeAbelianGroup.lift (Function. …
  -/
  apply lift.unique
  /-
    case hg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : AddCommGroup β
    inst✝ : AddCommGroup γ
    a : FreeAbelianGroup α
    f : α → β
    g : AddMonoidHom β γ
    ⊢ ∀ (x : α), Eq ((g.comp (FreeAbelianGroup.lift f)) (FreeAbelianGroup.of x)) ( …
  -/
  intro a
  /-
    case hg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : AddCommGroup β
    inst✝ : AddCommGroup γ
    a✝ : FreeAbelianGroup α
    f : α → β
    g : AddMonoidHom β γ
    a : α
    ⊢ Eq ((g.comp (FreeAbelianGroup.lift f)) (FreeAbelianGroup.of a)) (Function.co …
  -/
  show g ((lift f) (of a)) = g (f a)
  /-
    case hg
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : AddCommGroup β
    inst✝ : AddCommGroup γ
    a✝ : FreeAbelianGroup α
    f : α → β
    g : AddMonoidHom β γ
    a : α
    ⊢ Eq (g ((FreeAbelianGroup.lift f) (FreeAbelianGroup.of a))) (g (f a))
  -/
  simp only [(· ∘ ·), lift.of]
  /-
    🎉 no goals
  -/


theorem of_injective : Function.Injective (of : α → FreeAbelianGroup α) :=
  fun x y hoxy ↦ Classical.by_contradiction fun hxy : x ≠ y ↦
    let f : FreeAbelianGroup α →+ ℤ := lift fun z ↦ if x = z then (1 : ℤ) else 0
    have hfx1 : f (of x) = 1 := (lift.of _ _).trans <| if_pos rfl
    have hfy1 : f (of y) = 1 := hoxy ▸ hfx1
    have hfy0 : f (of y) = 0 := (lift.of _ _).trans <| if_neg hxy
    one_ne_zero <| hfy1.symm.trans hfy0


@[simp]
theorem of_ne_zero (x : α) : of x ≠ 0 := by
  /-
    α : Type u
    x : α
    ⊢ Ne (FreeAbelianGroup.of x) 0
  -/
  intro h
  /-
    α : Type u
    x : α
    h : Eq (FreeAbelianGroup.of x) 0
    ⊢ False
  -/
  let f : FreeAbelianGroup α →+ ℤ := lift 1
  /-
    α : Type u
    x : α
    h : Eq (FreeAbelianGroup.of x) 0
    f : AddMonoidHom (FreeAbelianGroup α) Int := FreeAbelianGroup.lift 1
    ⊢ False
  -/
  have hfx : f (of x) = 1 := lift.of _ _
  /-
    α : Type u
    x : α
    h : Eq (FreeAbelianGroup.of x) 0
    f : AddMonoidHom (FreeAbelianGroup α) Int := FreeAbelianGroup.lift 1
    hfx : Eq (f (FreeAbelianGroup.of x)) 1
    ⊢ False
  -/
  have hf0 : f (of x) = 0 := by rw [h, map_zero]
  /-
    α : Type u
    x : α
    h : Eq (FreeAbelianGroup.of x) 0
    f : AddMonoidHom (FreeAbelianGroup α) Int := FreeAbelianGroup.lift 1
    hfx : Eq (f (FreeAbelianGroup.of x)) 1
    hf0 : Eq (f (FreeAbelianGroup.of x)) 0
    ⊢ False
  -/
  exact one_ne_zero <| hfx.symm.trans hf0
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_ne_of (x : α) : 0 ≠ of x := of_ne_zero _ |>.symm


instance [Nonempty α] : Nontrivial (FreeAbelianGroup α) where
  exists_pair_ne := let ⟨x⟩ := ‹Nonempty α›; ⟨0, of x, zero_ne_of _⟩


@[elab_as_elim]
protected theorem induction_on {C : FreeAbelianGroup α → Prop} (z : FreeAbelianGroup α) (C0 : C 0)
    (C1 : ∀ x, C <| of x) (Cn : ∀ x, C (of x) → C (-of x)) (Cp : ∀ x y, C x → C y → C (x + y)) :
    C z :=
  Quotient.inductionOn' z fun x ↦
    Quot.inductionOn x fun L ↦
      List.recOn L C0 fun ⟨x, b⟩ _ ih ↦ Bool.recOn b (Cp _ _ (Cn _ (C1 x)) ih) (Cp _ _ (C1 x) ih)


theorem lift.add' {α β} [AddCommGroup β] (a : FreeAbelianGroup α) (f g : α → β) :
    lift (f + g) a = lift f a + lift g a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : AddCommGroup β
    a : FreeAbelianGroup α
    f g : α → β
    ⊢ Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) a) (HAdd.hAdd ((FreeAbelianGroup …
  -/
  refine FreeAbelianGroup.induction_on a ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      ⊢ Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) 0) (HAdd.hAdd ((FreeAbelianGroup …
    -/
  · simp only [(lift _).map_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (FreeAbelianGroup.of  …
    -/
  · intro x
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      x : α
      ⊢ Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (FreeAbelianGroup.of x)) (HAdd.h …
    -/
    simp only [lift.of, Pi.add_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (FreeAbelianGroup.of  …
    -/
  · intro x _
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      x : α
      a✝ : Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (FreeAbelianGroup.of x)) (HAd …
      ⊢ Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (Neg.neg (FreeAbelianGroup.of x) …
    -/
    simp only [map_neg, lift.of, Pi.add_apply, neg_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      ⊢ ∀ (x y : FreeAbelianGroup α), Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) x) …
    -/
  · intro x y hx hy
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      inst✝ : AddCommGroup β
      a : FreeAbelianGroup α
      f g : α → β
      x y : FreeAbelianGroup α
      hx : Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) x) (HAdd.hAdd ((FreeAbelianGr …
      hy : Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) y) (HAdd.hAdd ((FreeAbelianGr …
      ⊢ Eq ((FreeAbelianGroup.lift (HAdd.hAdd f g)) (HAdd.hAdd x y)) (HAdd.hAdd ((Fr …
    -/
    simp only [(lift _).map_add, hx, hy, add_add_add_comm]
    /-
      🎉 no goals
    -/


/-- If `g : FreeAbelianGroup X` and `A` is an abelian group then `liftAddGroupHom g`
is the additive group homomorphism sending a function `X → A` to the term of type `A`
corresponding to the evaluation of the induced map `FreeAbelianGroup X → A` at `g`. -/
@[simps!]  -- Porting note: Changed `simps` to `simps!`.
def liftAddGroupHom {α} (β) [AddCommGroup β] (a : FreeAbelianGroup α) : (α → β) →+ β :=
  AddMonoidHom.mk' (fun f ↦ lift f a) (lift.add' a)


theorem lift_neg' {β} [AddCommGroup β] (f : α → β) : lift (-f) = -lift f :=
  AddMonoidHom.ext fun _ ↦ (liftAddGroupHom _ _ : (α → β) →+ β).map_neg _


instance : Monad FreeAbelianGroup.{u} where
  pure α := of α
  bind x f := lift f x


@[elab_as_elim]
protected theorem induction_on' {C : FreeAbelianGroup α → Prop} (z : FreeAbelianGroup α) (C0 : C 0)
    (C1 : ∀ x, C <| pure x) (Cn : ∀ x, C (pure x) → C (-pure x))
    (Cp : ∀ x y, C x → C y → C (x + y)) : C z :=
  FreeAbelianGroup.induction_on z C0 C1 Cn Cp


@[simp]
theorem map_pure (f : α → β) (x : α) : f <$> (pure x : FreeAbelianGroup α) = pure (f x) :=
  rfl


@[simp]
protected theorem map_zero (f : α → β) : f <$> (0 : FreeAbelianGroup α) = 0 :=
  (lift (of ∘ f)).map_zero


@[simp]
protected theorem map_add (f : α → β) (x y : FreeAbelianGroup α) :
    f <$> (x + y) = f <$> x + f <$> y :=
  (lift _).map_add _ _


@[simp]
protected theorem map_neg (f : α → β) (x : FreeAbelianGroup α) : f <$> (-x) = -f <$> x :=
  map_neg (lift <| of ∘ f) _


@[simp]
protected theorem map_sub (f : α → β) (x y : FreeAbelianGroup α) :
    f <$> (x - y) = f <$> x - f <$> y :=
  map_sub (lift <| of ∘ f) _ _


@[simp]
theorem map_of (f : α → β) (y : α) : f <$> of y = of (f y) :=
  rfl


theorem pure_bind (f : α → FreeAbelianGroup β) (x) : pure x >>= f = f x :=
  lift.of _ _


@[simp]
theorem zero_bind (f : α → FreeAbelianGroup β) : 0 >>= f = 0 :=
  (lift f).map_zero


@[simp]
theorem add_bind (f : α → FreeAbelianGroup β) (x y : FreeAbelianGroup α) :
    x + y >>= f = (x >>= f) + (y >>= f) :=
  (lift _).map_add _ _


@[simp]
theorem neg_bind (f : α → FreeAbelianGroup β) (x : FreeAbelianGroup α) : -x >>= f = -(x >>= f) :=
  map_neg (lift f) _


@[simp]
theorem sub_bind (f : α → FreeAbelianGroup β) (x y : FreeAbelianGroup α) :
    x - y >>= f = (x >>= f) - (y >>= f) :=
  map_sub (lift f) _ _


@[simp]
theorem pure_seq (f : α → β) (x : FreeAbelianGroup α) : pure f <*> x = f <$> x :=
  pure_bind _ _


@[simp]
theorem zero_seq (x : FreeAbelianGroup α) : (0 : FreeAbelianGroup (α → β)) <*> x = 0 :=
  zero_bind _


@[simp]
theorem add_seq (f g : FreeAbelianGroup (α → β)) (x : FreeAbelianGroup α) :
    f + g <*> x = (f <*> x) + (g <*> x) :=
  add_bind _ _ _


@[simp]
theorem neg_seq (f : FreeAbelianGroup (α → β)) (x : FreeAbelianGroup α) : -f <*> x = -(f <*> x) :=
  neg_bind _ _


@[simp]
theorem sub_seq (f g : FreeAbelianGroup (α → β)) (x : FreeAbelianGroup α) :
    f - g <*> x = (f <*> x) - (g <*> x) :=
  sub_bind _ _ _


/-- If `f : FreeAbelianGroup (α → β)`, then `f <*>` is an additive morphism
`FreeAbelianGroup α →+ FreeAbelianGroup β`. -/
def seqAddGroupHom (f : FreeAbelianGroup (α → β)) : FreeAbelianGroup α →+ FreeAbelianGroup β :=
  AddMonoidHom.mk' (f <*> ·) fun x y ↦
    show lift (· <$> (x + y)) _ = _ by
      /-
        α β : Type u
        f : FreeAbelianGroup (α → β)
        x y : FreeAbelianGroup α
        ⊢ Eq ((FreeAbelianGroup.lift fun x_1 => Functor.map x_1 (HAdd.hAdd x y)) f) (H …
      -/
      simp only [FreeAbelianGroup.map_add]
      /-
        α β : Type u
        f : FreeAbelianGroup (α → β)
        x y : FreeAbelianGroup α
        ⊢ Eq ((FreeAbelianGroup.lift fun x_1 => HAdd.hAdd (Functor.map x_1 x) (Functor …
      -/
      exact lift.add' f _ _
      /-
        🎉 no goals
      -/


@[simp]
theorem seq_zero (f : FreeAbelianGroup (α → β)) : f <*> 0 = 0 :=
  (seqAddGroupHom f).map_zero


@[simp]
theorem seq_add (f : FreeAbelianGroup (α → β)) (x y : FreeAbelianGroup α) :
    f <*> x + y = (f <*> x) + (f <*> y) :=
  (seqAddGroupHom f).map_add x y


@[simp]
theorem seq_neg (f : FreeAbelianGroup (α → β)) (x : FreeAbelianGroup α) : f <*> -x = -(f <*> x) :=
  (seqAddGroupHom f).map_neg x


@[simp]
theorem seq_sub (f : FreeAbelianGroup (α → β)) (x y : FreeAbelianGroup α) :
    f <*> x - y = (f <*> x) - (f <*> y) :=
  (seqAddGroupHom f).map_sub x y


                                               /-
                                                 α β : Type u
                                                 ⊢ ∀ {α β : Type u} (x : α) (y : FreeAbelianGroup β), Eq (Functor.mapConst x y) …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                   /-
                     α β α✝ : Type u
                     x✝ : FreeAbelianGroup α✝
                     x : α✝
                     ih : Eq (Functor.map id (Pure.pure x)) (Pure.pure x)
                     ⊢ Eq (Functor.map id (Neg.neg (Pure.pure x))) (Neg.neg (Pure.pure x))
                   -/
                                               /-
                                                 🎉 no goals
                                               -/
                   /-
                     🎉 no goals
                   -/
                         /-
                           α β α✝ : Type u
                           x✝ x y : FreeAbelianGroup α✝
                           ihx : Eq (Functor.map id x) x
                           ihy : Eq (Functor.map id y) y
                           ⊢ Eq (Functor.map id (HAdd.hAdd x y)) (HAdd.hAdd x y)
                         -/
                                               /-
                                                 🎉 no goals
                                               -/
                         /-
                           🎉 no goals
                         -/
                                               /-
                                                 🎉 no goals
                                               -/
                                                                  /-
                                                                    α β α✝ β✝ γ✝ : Type u
                                                                    x : FreeAbelianGroup α✝
                                                                    f : α✝ → FreeAbelianGroup β✝
                                                                    g : β✝ → FreeAbelianGroup γ✝
                                                                    ⊢ Eq (Bind.bind (Bind.bind 0 f) g) (Bind.bind 0 fun x => Bind.bind (f x) g)
                                                                  -/
instance : LawfulMonad FreeAbelianGroup.{u} := LawfulMonad.mk'
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                /-
                  α β α✝ β✝ γ✝ : Type u
                  x✝ : FreeAbelianGroup α✝
                  f : α✝ → FreeAbelianGroup β✝
                  g : β✝ → FreeAbelianGroup γ✝
                  x : α✝
                  ⊢ Eq (Bind.bind (Bind.bind (Pure.pure x) f) g) (Bind.bind (Pure.pure x) fun x  …
                -/
                /-
                  🎉 no goals
                -/
                                               /-
                                                 🎉 no goals
                                               -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                         /-
                           α β α✝ β✝ γ✝ : Type u
                           x✝ : FreeAbelianGroup α✝
                           f : α✝ → FreeAbelianGroup β✝
                           g : β✝ → FreeAbelianGroup γ✝
                           x y : FreeAbelianGroup α✝
                           ihx : Eq (Bind.bind (Bind.bind x f) g) (Bind.bind x fun x => Bind.bind (f x) g)
                           ihy : Eq (Bind.bind (Bind.bind y f) g) (Bind.bind y fun x => Bind.bind (f x) g)
                           ⊢ Eq (Bind.bind (Bind.bind (HAdd.hAdd x y) f) g) (Bind.bind (HAdd.hAdd x y) fu …
                         -/
  (id_map := fun x ↦ FreeAbelianGroup.induction_on' x (FreeAbelianGroup.map_zero id) (map_pure id)
                         /-
                           🎉 no goals
                         -/
    (fun x ih ↦ by rw [FreeAbelianGroup.map_neg, ih])
    fun x y ihx ihy ↦ by rw [FreeAbelianGroup.map_add, ihx, ihy])
  (pure_bind := fun x f ↦ pure_bind f x)
  (bind_assoc := fun x f g ↦ FreeAbelianGroup.induction_on' x (by iterate 3 rw [zero_bind])
    (fun x ↦ by iterate 2 rw [pure_bind]) (fun x ih ↦ by iterate 3 rw [neg_bind] <;> try rw [ih])
    fun x y ihx ihy ↦ by iterate 3 rw [add_bind] <;> try rw [ihx, ihy])


instance : CommApplicative FreeAbelianGroup.{u} where
  commutative_prod x y := by
    /-
      α β α✝ β✝ : Type u
      x : FreeAbelianGroup α✝
      y : FreeAbelianGroup β✝
      ⊢ Eq (Seq.seq (Functor.map Prod.mk x) fun x => y) (Seq.seq (Functor.map (fun b …
    -/
    refine FreeAbelianGroup.induction_on' x ?_ ?_ ?_ ?_
      /-
        case refine_1
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        ⊢ Eq (Seq.seq (Functor.map Prod.mk 0) fun x => y) (Seq.seq (Functor.map (fun b …
      -/
    · rw [FreeAbelianGroup.map_zero, zero_seq, seq_zero]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        ⊢ ∀ (x : α✝), Eq (Seq.seq (Functor.map Prod.mk (Pure.pure x)) fun x => y) (Seq …
      -/
    · intro p
      /-
        case refine_2
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        p : α✝
        ⊢ Eq (Seq.seq (Functor.map Prod.mk (Pure.pure p)) fun x => y) (Seq.seq (Functo …
      -/
      rw [map_pure, pure_seq]
      exact FreeAbelianGroup.induction_on' y
        (by rw [FreeAbelianGroup.map_zero, FreeAbelianGroup.map_zero, zero_seq])
        (fun q ↦ by rw [map_pure, map_pure, pure_seq, map_pure])
        (fun q ih ↦ by rw [FreeAbelianGroup.map_neg, FreeAbelianGroup.map_neg, neg_seq, ih])
        fun y₁ y₂ ih1 ih2 ↦ by
          rw [FreeAbelianGroup.map_add, FreeAbelianGroup.map_add, add_seq, ih1, ih2]
      /-
        case refine_3
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        ⊢ ∀ (x : α✝), Eq (Seq.seq (Functor.map Prod.mk (Pure.pure x)) fun x => y) (Seq …
      -/
    · intro p ih
      /-
        case refine_3
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        p : α✝
        ih : Eq (Seq.seq (Functor.map Prod.mk (Pure.pure p)) fun x => y) (Seq.seq (Fun …
        ⊢ Eq (Seq.seq (Functor.map Prod.mk (Neg.neg (Pure.pure p))) fun x => y) (Seq.s …
      -/
      rw [FreeAbelianGroup.map_neg, neg_seq, seq_neg, ih]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        ⊢ ∀ (x y_1 : FreeAbelianGroup α✝), Eq (Seq.seq (Functor.map Prod.mk x) fun x = …
      -/
    · intro x₁ x₂ ih1 ih2
      /-
        case refine_4
        α β α✝ β✝ : Type u
        x : FreeAbelianGroup α✝
        y : FreeAbelianGroup β✝
        x₁ x₂ : FreeAbelianGroup α✝
        ih1 : Eq (Seq.seq (Functor.map Prod.mk x₁) fun x => y) (Seq.seq (Functor.map ( …
        ih2 : Eq (Seq.seq (Functor.map Prod.mk x₂) fun x => y) (Seq.seq (Functor.map ( …
        ⊢ Eq (Seq.seq (Functor.map Prod.mk (HAdd.hAdd x₁ x₂)) fun x => y) (Seq.seq (Fu …
      -/
      rw [FreeAbelianGroup.map_add, add_seq, seq_add, ih1, ih2]
      /-
        🎉 no goals
      -/


/-- The additive group homomorphism `FreeAbelianGroup α →+ FreeAbelianGroup β` induced from a
  map `α → β`. -/
def map (f : α → β) : FreeAbelianGroup α →+ FreeAbelianGroup β :=
  lift (of ∘ f)


theorem lift_comp {α} {β} {γ} [AddCommGroup γ] (f : α → β) (g : β → γ) (x : FreeAbelianGroup α) :
    lift (g ∘ f) x = lift g (map f x) := by
  -- Porting note: Added motive.
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : AddCommGroup γ
    f : α → β
    g : β → γ
    x : FreeAbelianGroup α
    ⊢ Eq ((FreeAbelianGroup.lift (Function.comp g f)) x) ((FreeAbelianGroup.lift g …
  -/
  apply FreeAbelianGroup.induction_on (C := fun x ↦ lift (g ∘ f) x = lift g (map f x)) x
    /-
      case C0
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      ⊢ Eq ((FreeAbelianGroup.lift (Function.comp g f)) 0) ((FreeAbelianGroup.lift g …
    -/
  · simp only [map_zero]
    /-
      🎉 no goals
    -/
    /-
      case C1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift (Function.comp g f)) (FreeAbelianGroup …
    -/
  · intro _
    /-
      case C1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      x✝ : α
      ⊢ Eq ((FreeAbelianGroup.lift (Function.comp g f)) (FreeAbelianGroup.of x✝)) (( …
    -/
    simp only [lift.of, map, Function.comp]
    /-
      🎉 no goals
    -/
    /-
      case Cn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift (Function.comp g f)) (FreeAbelianGroup …
    -/
  · intro _ h
    /-
      case Cn
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      x✝ : α
      h : Eq ((FreeAbelianGroup.lift (Function.comp g f)) (FreeAbelianGroup.of x✝))  …
      ⊢ Eq ((FreeAbelianGroup.lift (Function.comp g f)) (Neg.neg (FreeAbelianGroup.o …
    -/
    simp only [h, AddMonoidHom.map_neg]
    /-
      🎉 no goals
    -/
    /-
      case Cp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x : FreeAbelianGroup α
      ⊢ ∀ (x y : FreeAbelianGroup α), Eq ((FreeAbelianGroup.lift (Function.comp g f) …
    -/
  · intro _ _ h₁ h₂
    /-
      case Cp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : AddCommGroup γ
      f : α → β
      g : β → γ
      x x✝ y✝ : FreeAbelianGroup α
      h₁ : Eq ((FreeAbelianGroup.lift (Function.comp g f)) x✝) ((FreeAbelianGroup.li …
      h₂ : Eq ((FreeAbelianGroup.lift (Function.comp g f)) y✝) ((FreeAbelianGroup.li …
      ⊢ Eq ((FreeAbelianGroup.lift (Function.comp g f)) (HAdd.hAdd x✝ y✝)) ((FreeAbe …
    -/
    simp only [h₁, h₂, AddMonoidHom.map_add]
    /-
      🎉 no goals
    -/


theorem map_id : map id = AddMonoidHom.id (FreeAbelianGroup α) :=
  Eq.symm <|
    lift.ext _ _ fun _ ↦ lift.unique of (AddMonoidHom.id _) fun _ ↦ AddMonoidHom.id_apply _ _


theorem map_id_apply (x : FreeAbelianGroup α) : map id x = x := by
  /-
    α : Type u
    x : FreeAbelianGroup α
    ⊢ Eq ((FreeAbelianGroup.map id) x) x
  -/
  rw [map_id]
  /-
    α : Type u
    x : FreeAbelianGroup α
    ⊢ Eq ((AddMonoidHom.id (FreeAbelianGroup α)) x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_comp {f : α → β} {g : β → γ} : map (g ∘ f) = (map g).comp (map f) :=
                                     /-
                                       α : Type u
                                       β : Type v
                                       γ : Type w
                                       f : α → β
                                       g : β → γ
                                       x✝ : α
                                       ⊢ Eq (((FreeAbelianGroup.map g).comp (FreeAbelianGroup.map f)) (FreeAbelianGro …
                                     -/
  Eq.symm <| lift.ext _ _ fun _ ↦ by simp [map]
                                     /-
                                       🎉 no goals
                                     -/


theorem map_comp_apply {f : α → β} {g : β → γ} (x : FreeAbelianGroup α) :
    map (g ∘ f) x = (map g) ((map f) x) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    x : FreeAbelianGroup α
    ⊢ Eq ((FreeAbelianGroup.map (Function.comp g f)) x) ((FreeAbelianGroup.map g)  …
  -/
  rw [map_comp]
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    x : FreeAbelianGroup α
    ⊢ Eq (((FreeAbelianGroup.map g).comp (FreeAbelianGroup.map f)) x) ((FreeAbelia …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- version of map_of which uses `map`

@[simp]
theorem map_of_apply {f : α → β} (a : α) : map f (of a) = of (f a) :=
  rfl


instance mul : Mul (FreeAbelianGroup α) :=
  ⟨fun x ↦ lift fun x₂ ↦ lift (fun x₁ ↦ of (x₁ * x₂)) x⟩


theorem mul_def (x y : FreeAbelianGroup α) :
    x * y = lift (fun x₂ ↦ lift (fun x₁ ↦ of (x₁ * x₂)) x) y :=
  rfl


@[simp]
theorem of_mul_of (x y : α) : of x * of y = of (x * y) := by
  /-
    α : Type u
    inst✝ : Mul α
    x y : α
    ⊢ Eq (HMul.hMul (FreeAbelianGroup.of x) (FreeAbelianGroup.of y)) (FreeAbelianG …
  -/
  rw [mul_def, lift.of, lift.of]
  /-
    🎉 no goals
  -/


theorem of_mul (x y : α) : of (x * y) = of x * of y :=
  Eq.symm <| of_mul_of x y


instance distrib : Distrib (FreeAbelianGroup α) :=
  { FreeAbelianGroup.mul α, FreeAbelianGroup.addCommGroup α with
    left_distrib := fun _ _ _ ↦ (lift _).map_add _ _
                                    /-
                                      α : Type u
                                      β : Type v
                                      γ : Type w
                                      inst✝ : Mul α
                                      x y z : FreeAbelianGroup α
                                      ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
                                    -/
    right_distrib := fun x y z ↦ by simp only [(· * ·), Mul.mul, map_add, ← Pi.add_def, lift.add'] }
                                    /-
                                      🎉 no goals
                                    -/


instance nonUnitalNonAssocRing : NonUnitalNonAssocRing (FreeAbelianGroup α) :=
  { FreeAbelianGroup.distrib,
    FreeAbelianGroup.addCommGroup _ with
    zero_mul := fun a ↦ by
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝ : Mul α
        a : FreeAbelianGroup α
        ⊢ Eq (HMul.hMul 0 a) 0
      -/
      have h : 0 * a + 0 * a = 0 * a := by simp [← add_mul]
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝ : Mul α
        a : FreeAbelianGroup α
        h : Eq (HAdd.hAdd (HMul.hMul 0 a) (HMul.hMul 0 a)) (HMul.hMul 0 a)
        ⊢ Eq (HMul.hMul 0 a) 0
      -/
      simpa using h
      /-
        🎉 no goals
      -/
    mul_zero := fun _ ↦ rfl }


instance one : One (FreeAbelianGroup α) :=
  ⟨of 1⟩


theorem one_def : (1 : FreeAbelianGroup α) = of 1 :=
  rfl


theorem of_one : (of 1 : FreeAbelianGroup α) = 1 :=
  rfl


instance nonUnitalRing [Semigroup α] : NonUnitalRing (FreeAbelianGroup α) :=
  { FreeAbelianGroup.nonUnitalNonAssocRing with
    mul_assoc := fun x y z ↦ by
      refine FreeAbelianGroup.induction_on z (by simp only [mul_zero])
          (fun L3 ↦ ?_) (fun L3 ih ↦ ?_) fun z₁ z₂ ih₁ ih₂ ↦ ?_
      · refine FreeAbelianGroup.induction_on y (by simp only [mul_zero, zero_mul])
            (fun L2 ↦ ?_) (fun L2 ih ↦ ?_) fun y₁ y₂ ih₁ ih₂ ↦ ?_
        · refine FreeAbelianGroup.induction_on x (by simp only [zero_mul])
              (fun L1 ↦ ?_) (fun L1 ih ↦ ?_) fun x₁ x₂ ih₁ ih₂ ↦ ?_
            /-
              case refine_1.refine_1.refine_1
              α : Type u
              β : Type v
              γ : Type w
              inst✝ : Semigroup α
              x y z : FreeAbelianGroup α
              L3 L2 L1 : α
              ⊢ Eq (HMul.hMul (HMul.hMul (FreeAbelianGroup.of L1) (FreeAbelianGroup.of L2))  …
            -/
          · rw [of_mul_of, of_mul_of, of_mul_of, of_mul_of, mul_assoc]
            /-
              🎉 no goals
            -/
            /-
              case refine_1.refine_1.refine_2
              α : Type u
              β : Type v
              γ : Type w
              inst✝ : Semigroup α
              x y z : FreeAbelianGroup α
              L3 L2 L1 : α
              ih : Eq (HMul.hMul (HMul.hMul (FreeAbelianGroup.of L1) (FreeAbelianGroup.of L2 …
              ⊢ Eq (HMul.hMul (HMul.hMul (Neg.neg (FreeAbelianGroup.of L1)) (FreeAbelianGrou …
            -/
          · rw [neg_mul, neg_mul, neg_mul, ih]
            /-
              🎉 no goals
            -/
            /-
              case refine_1.refine_1.refine_3
              α : Type u
              β : Type v
              γ : Type w
              inst✝ : Semigroup α
              x y z : FreeAbelianGroup α
              L3 L2 : α
              x₁ x₂ : FreeAbelianGroup α
              ih₁ : Eq (HMul.hMul (HMul.hMul x₁ (FreeAbelianGroup.of L2)) (FreeAbelianGroup. …
              ih₂ : Eq (HMul.hMul (HMul.hMul x₂ (FreeAbelianGroup.of L2)) (FreeAbelianGroup. …
              ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd x₁ x₂) (FreeAbelianGroup.of L2)) (FreeAb …
            -/
          · rw [add_mul, add_mul, add_mul, ih₁, ih₂]
            /-
              🎉 no goals
            -/
          /-
            case refine_1.refine_2
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : Semigroup α
            x y z : FreeAbelianGroup α
            L3 L2 : α
            ih : Eq (HMul.hMul (HMul.hMul x (FreeAbelianGroup.of L2)) (FreeAbelianGroup.of …
            ⊢ Eq (HMul.hMul (HMul.hMul x (Neg.neg (FreeAbelianGroup.of L2))) (FreeAbelianG …
          -/
        · rw [neg_mul, mul_neg, mul_neg, neg_mul, ih]
          /-
            🎉 no goals
          -/
          /-
            case refine_1.refine_3
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : Semigroup α
            x y z : FreeAbelianGroup α
            L3 : α
            y₁ y₂ : FreeAbelianGroup α
            ih₁ : Eq (HMul.hMul (HMul.hMul x y₁) (FreeAbelianGroup.of L3)) (HMul.hMul x (H …
            ih₂ : Eq (HMul.hMul (HMul.hMul x y₂) (FreeAbelianGroup.of L3)) (HMul.hMul x (H …
            ⊢ Eq (HMul.hMul (HMul.hMul x (HAdd.hAdd y₁ y₂)) (FreeAbelianGroup.of L3)) (HMu …
          -/
        · rw [add_mul, mul_add, mul_add, add_mul, ih₁, ih₂]
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : Semigroup α
          x y z : FreeAbelianGroup α
          L3 : α
          ih : Eq (HMul.hMul (HMul.hMul x y) (FreeAbelianGroup.of L3)) (HMul.hMul x (HMu …
          ⊢ Eq (HMul.hMul (HMul.hMul x y) (Neg.neg (FreeAbelianGroup.of L3))) (HMul.hMul …
        -/
      · rw [mul_neg, mul_neg, mul_neg, ih]
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : Semigroup α
          x y z z₁ z₂ : FreeAbelianGroup α
          ih₁ : Eq (HMul.hMul (HMul.hMul x y) z₁) (HMul.hMul x (HMul.hMul y z₁))
          ih₂ : Eq (HMul.hMul (HMul.hMul x y) z₂) (HMul.hMul x (HMul.hMul y z₂))
          ⊢ Eq (HMul.hMul (HMul.hMul x y) (HAdd.hAdd z₁ z₂)) (HMul.hMul x (HMul.hMul y ( …
        -/
      · rw [mul_add, mul_add, mul_add, ih₁, ih₂] }
        /-
          🎉 no goals
        -/


instance ring : Ring (FreeAbelianGroup α) :=
  { FreeAbelianGroup.nonUnitalRing _,
    FreeAbelianGroup.one _ with
    mul_one := fun x ↦ by
      /-
        α : Type u
        β : Type v
        γ : Type w
        R : Type u_1
        inst✝¹ : Monoid α
        inst✝ : Ring R
        x : FreeAbelianGroup α
        ⊢ Eq (HMul.hMul x 1) x
      -/
      rw [mul_def, one_def, lift.of]
      /-
        α : Type u
        β : Type v
        γ : Type w
        R : Type u_1
        inst✝¹ : Monoid α
        inst✝ : Ring R
        x : FreeAbelianGroup α
        ⊢ Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1)) x …
      -/
      refine FreeAbelianGroup.induction_on x rfl (fun L ↦ ?_) (fun L ih ↦ ?_) fun x1 x2 ih1 ih2 ↦ ?_
        /-
          case refine_1
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          L : α
          ⊢ Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1)) ( …
        -/
      · rw [lift.of, mul_one]
      /-
        α : Type u
        β : Type v
        γ : Type w
        R : Type u_1
        inst✝¹ : Monoid α
        inst✝ : Ring R
        x : FreeAbelianGroup α
        ⊢ Eq (HMul.hMul 1 x) x
      -/
        /-
          🎉 no goals
        -/
      /-
        α : Type u
        β : Type v
        γ : Type w
        R : Type u_1
        inst✝¹ : Monoid α
        inst✝ : Ring R
        x : FreeAbelianGroup α
        ⊢ Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂)) x …
      -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          L : α
          ih : Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1) …
          ⊢ Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1)) ( …
        -/
        /-
          case refine_1
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hM …
        -/
      · rw [map_neg, ih]
        /-
          case refine_1
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          L : α
          ⊢ Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂)) ( …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          ⊢ ∀ (x : α), Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hM …
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x x1 x2 : FreeAbelianGroup α
          ih1 : Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1 …
          ih2 : Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1 …
          ⊢ Eq ((FreeAbelianGroup.lift fun x₁ => FreeAbelianGroup.of (HMul.hMul x₁ 1)) ( …
        -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          L : α
          ih : Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂) …
          ⊢ Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂)) ( …
        -/
      · rw [map_add, ih1, ih2]
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x : FreeAbelianGroup α
          ⊢ ∀ (x y : FreeAbelianGroup α), Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbeli …
        -/
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          x x1 x2 : FreeAbelianGroup α
          ih1 : Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂ …
          ih2 : Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂ …
          ⊢ Eq ((FreeAbelianGroup.lift fun x₂ => FreeAbelianGroup.of (HMul.hMul 1 x₂)) ( …
        -/
    one_mul := fun x ↦ by
        /-
          🎉 no goals
        -/
      simp_rw [mul_def, one_def, lift.of]
      refine FreeAbelianGroup.induction_on x rfl ?_ ?_ ?_
      · intro L
        rw [lift.of, one_mul]
      · intro L ih
        rw [map_neg, ih]
      · intro x1 x2 ih1 ih2
        rw [map_add, ih1, ih2] }


/-- `FreeAbelianGroup.of` is a `MonoidHom` when `α` is a `Monoid`. -/
def ofMulHom : α →* FreeAbelianGroup α where
  toFun := of
  map_one' := of_one _
  map_mul' := of_mul


@[simp]
theorem ofMulHom_coe : (ofMulHom : α → FreeAbelianGroup α) = of :=
  rfl


/-- If `f` preserves multiplication, then so does `lift f`. -/
def liftMonoid : (α →* R) ≃ (FreeAbelianGroup α →+* R) where
  toFun f := { lift f with
    toFun := lift f
    map_one' := (lift.of f _).trans f.map_one
    map_mul' := fun x y ↦ by
      /-
        α : Type u
        β : Type v
        γ : Type w
        R : Type u_1
        inst✝¹ : Monoid α
        inst✝ : Ring R
        f : MonoidHom α R
        x y : FreeAbelianGroup α
        ⊢ Eq ({ toFun := ⇑(FreeAbelianGroup.lift ⇑f), map_one' := ⋯ }.toFun (HMul.hMul …
      -/
      simp only
      refine FreeAbelianGroup.induction_on y
          (by simp only [mul_zero, map_zero]) (fun L2 ↦ ?_) (fun L2 ih ↦ ?_) ?_
      · refine FreeAbelianGroup.induction_on x
            (by simp only [zero_mul, map_zero]) (fun L1 ↦ ?_) (fun L1 ih ↦ ?_) ?_
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 L1 : α
            ⊢ Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (FreeAbelianGroup.of L1) (FreeAbel …
          -/
        · simp_rw [of_mul_of, lift.of]
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 L1 : α
            ⊢ Eq (f (HMul.hMul L1 L2)) (HMul.hMul (f L1) (f L2))
          -/
          exact f.map_mul _ _
          /-
            🎉 no goals
          -/
          /-
            case refine_1.refine_2
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 L1 : α
            ih : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (FreeAbelianGroup.of L1) (FreeA …
            ⊢ Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (Neg.neg (FreeAbelianGroup.of L1)) …
          -/
        · simp_rw [neg_mul, map_neg, neg_mul]
          /-
            case refine_1.refine_2
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 L1 : α
            ih : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (FreeAbelianGroup.of L1) (FreeA …
            ⊢ Eq (Neg.neg ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (FreeAbelianGroup.of L1)  …
          -/
          exact congr_arg Neg.neg ih
          /-
            🎉 no goals
          -/
          /-
            case refine_1.refine_3
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 : α
            ⊢ ∀ (x y : FreeAbelianGroup α), Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x (F …
          -/
        · intro x1 x2 ih1 ih2
          /-
            case refine_1.refine_3
            α : Type u
            β : Type v
            γ : Type w
            R : Type u_1
            inst✝¹ : Monoid α
            inst✝ : Ring R
            f : MonoidHom α R
            x y : FreeAbelianGroup α
            L2 : α
            x1 x2 : FreeAbelianGroup α
            ih1 : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x1 (FreeAbelianGroup.of L2)))  …
            ih2 : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x2 (FreeAbelianGroup.of L2)))  …
            ⊢ Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul (HAdd.hAdd x1 x2) (FreeAbelianGrou …
          -/
          simp only [add_mul, map_add, ih1, ih2]
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          f : MonoidHom α R
          x y : FreeAbelianGroup α
          L2 : α
          ih : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x (FreeAbelianGroup.of L2))) (H …
          ⊢ Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x (Neg.neg (FreeAbelianGroup.of L2 …
        -/
      · rw [mul_neg, map_neg, map_neg, mul_neg, ih]
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          f : MonoidHom α R
          x y : FreeAbelianGroup α
          ⊢ ∀ (x_1 y : FreeAbelianGroup α), Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x  …
        -/
      · intro y1 y2 ih1 ih2
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          R : Type u_1
          inst✝¹ : Monoid α
          inst✝ : Ring R
          f : MonoidHom α R
          x y y1 y2 : FreeAbelianGroup α
          ih1 : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x y1)) (HMul.hMul ((FreeAbelia …
          ih2 : Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x y2)) (HMul.hMul ((FreeAbelia …
          ⊢ Eq ((FreeAbelianGroup.lift ⇑f) (HMul.hMul x (HAdd.hAdd y1 y2))) (HMul.hMul ( …
        -/
        rw [mul_add, map_add, map_add, mul_add, ih1, ih2] }
        /-
          🎉 no goals
        -/
  invFun F := MonoidHom.comp (↑F) ofMulHom
  left_inv f := MonoidHom.ext <| by
    simp only [RingHom.coe_monoidHom_mk, MonoidHom.coe_comp, MonoidHom.coe_mk, OneHom.coe_mk,
      ofMulHom_coe, Function.comp_apply, lift.of, forall_const]
  right_inv F := RingHom.coe_addMonoidHom_injective <| by
    /-
      α : Type u
      β : Type v
      γ : Type w
      R : Type u_1
      inst✝¹ : Monoid α
      inst✝ : Ring R
      F : RingHom (FreeAbelianGroup α) R
      ⊢ Eq
          ((fun f => ↑f)
            ((fun f =>
                let __src := FreeAbelianGroup.lift ⇑f;
                { toFun := ⇑(FreeAbelianGroup.lift ⇑f), map_one' := ⋯, map_mul' := ⋯ …
              ((fun F => (↑F).comp FreeAbelianGroup.ofMulHom) F)))
          ((fun f => ↑f) F)
    -/
    simp only
    /-
      α : Type u
      β : Type v
      γ : Type w
      R : Type u_1
      inst✝¹ : Monoid α
      inst✝ : Ring R
      F : RingHom (FreeAbelianGroup α) R
      ⊢ Eq ↑{ toFun := ⇑(FreeAbelianGroup.lift ⇑((↑F).comp FreeAbelianGroup.ofMulHom …
    -/
    rw [← lift.apply_symm_apply (↑F : FreeAbelianGroup α →+ R)]
    /-
      α : Type u
      β : Type v
      γ : Type w
      R : Type u_1
      inst✝¹ : Monoid α
      inst✝ : Ring R
      F : RingHom (FreeAbelianGroup α) R
      ⊢ Eq (↑{ toFun := ⇑(FreeAbelianGroup.lift ⇑((↑F).comp FreeAbelianGroup.ofMulHo …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem liftMonoid_coe_addMonoidHom (f : α →* R) : ↑(liftMonoid f) = lift f :=
  rfl


@[simp]
theorem liftMonoid_coe (f : α →* R) : ⇑(liftMonoid f) = lift f :=
  rfl


@[simp]
-- Porting note: Added a type to `↑f`.
theorem liftMonoid_symm_coe (f : FreeAbelianGroup α →+* R) :
    ⇑(liftMonoid.symm f) = lift.symm (↑f : FreeAbelianGroup α →+ R) :=
  rfl


instance [CommMonoid α] : CommRing (FreeAbelianGroup α) :=
  { FreeAbelianGroup.ring α with
    mul_comm := fun x y ↦ by
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝ : CommMonoid α
        x y : FreeAbelianGroup α
        ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
      -/
      refine FreeAbelianGroup.induction_on x (zero_mul y) ?_ ?_ ?_
        /-
          case refine_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y : FreeAbelianGroup α
          ⊢ ∀ (x : α), Eq (HMul.hMul (FreeAbelianGroup.of x) y) (HMul.hMul y (FreeAbelia …
        -/
      · intro s
        /-
          case refine_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y : FreeAbelianGroup α
          s : α
          ⊢ Eq (HMul.hMul (FreeAbelianGroup.of s) y) (HMul.hMul y (FreeAbelianGroup.of s))
        -/
        refine FreeAbelianGroup.induction_on y (zero_mul _).symm ?_ ?_ ?_
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s : α
            ⊢ ∀ (x : α), Eq (HMul.hMul (FreeAbelianGroup.of s) (FreeAbelianGroup.of x)) (H …
          -/
        · intro t
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s t : α
            ⊢ Eq (HMul.hMul (FreeAbelianGroup.of s) (FreeAbelianGroup.of t)) (HMul.hMul (F …
          -/
          dsimp only [(· * ·), Mul.mul]
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s t : α
            ⊢ Eq ((FreeAbelianGroup.lift fun x₂ => (FreeAbelianGroup.lift fun x₁ => FreeAb …
          -/
          iterate 4 rw [lift.of]
          /-
            case refine_1.refine_1
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s t : α
            ⊢ Eq (FreeAbelianGroup.of (Mul.mul s t)) (FreeAbelianGroup.of (Mul.mul t s))
          -/
          congr 1
          /-
            case refine_1.refine_1.e_x
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s t : α
            ⊢ Eq (Mul.mul s t) (Mul.mul t s)
          -/
          exact mul_comm _ _
          /-
            🎉 no goals
          -/
          /-
            case refine_1.refine_2
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s : α
            ⊢ ∀ (x : α), Eq (HMul.hMul (FreeAbelianGroup.of s) (FreeAbelianGroup.of x)) (H …
          -/
        · intro t ih
          /-
            case refine_1.refine_2
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s t : α
            ih : Eq (HMul.hMul (FreeAbelianGroup.of s) (FreeAbelianGroup.of t)) (HMul.hMul …
            ⊢ Eq (HMul.hMul (FreeAbelianGroup.of s) (Neg.neg (FreeAbelianGroup.of t))) (HM …
          -/
          rw [mul_neg, ih, neg_mul_eq_neg_mul]
          /-
            🎉 no goals
          -/
          /-
            case refine_1.refine_3
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s : α
            ⊢ ∀ (x y : FreeAbelianGroup α), Eq (HMul.hMul (FreeAbelianGroup.of s) x) (HMul …
          -/
        · intro y1 y2 ih1 ih2
          /-
            case refine_1.refine_3
            α : Type u
            β : Type v
            γ : Type w
            inst✝ : CommMonoid α
            x y : FreeAbelianGroup α
            s : α
            y1 y2 : FreeAbelianGroup α
            ih1 : Eq (HMul.hMul (FreeAbelianGroup.of s) y1) (HMul.hMul y1 (FreeAbelianGrou …
            ih2 : Eq (HMul.hMul (FreeAbelianGroup.of s) y2) (HMul.hMul y2 (FreeAbelianGrou …
            ⊢ Eq (HMul.hMul (FreeAbelianGroup.of s) (HAdd.hAdd y1 y2)) (HMul.hMul (HAdd.hA …
          -/
          rw [mul_add, add_mul, ih1, ih2]
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y : FreeAbelianGroup α
          ⊢ ∀ (x : α), Eq (HMul.hMul (FreeAbelianGroup.of x) y) (HMul.hMul y (FreeAbelia …
        -/
      · intro s ih
        /-
          case refine_2
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y : FreeAbelianGroup α
          s : α
          ih : Eq (HMul.hMul (FreeAbelianGroup.of s) y) (HMul.hMul y (FreeAbelianGroup.o …
          ⊢ Eq (HMul.hMul (Neg.neg (FreeAbelianGroup.of s)) y) (HMul.hMul y (Neg.neg (Fr …
        -/
        rw [neg_mul, ih, neg_mul_eq_mul_neg]
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y : FreeAbelianGroup α
          ⊢ ∀ (x y_1 : FreeAbelianGroup α), Eq (HMul.hMul x y) (HMul.hMul y x) → Eq (HMu …
        -/
      · intro x1 x2 ih1 ih2
        /-
          case refine_3
          α : Type u
          β : Type v
          γ : Type w
          inst✝ : CommMonoid α
          x y x1 x2 : FreeAbelianGroup α
          ih1 : Eq (HMul.hMul x1 y) (HMul.hMul y x1)
          ih2 : Eq (HMul.hMul x2 y) (HMul.hMul y x2)
          ⊢ Eq (HMul.hMul (HAdd.hAdd x1 x2) y) (HMul.hMul y (HAdd.hAdd x1 x2))
        -/
        rw [add_mul, mul_add, ih1, ih2] }
        /-
          🎉 no goals
        -/


instance pemptyUnique : Unique (FreeAbelianGroup PEmpty) where
  default := 0
  uniq x := FreeAbelianGroup.induction_on x rfl (PEmpty.elim ·) (PEmpty.elim ·) (by
    /-
      α : Type u
      β : Type v
      γ : Type w
      x : FreeAbelianGroup PEmpty.{?u.112426 + 1}
      ⊢ ∀ (x y : FreeAbelianGroup PEmpty.{?u.112426 + 1}), Eq x Inhabited.default →  …
    -/
    rintro - - rfl rfl
    /-
      α : Type u
      β : Type v
      γ : Type w
      x : FreeAbelianGroup PEmpty.{?u.112426 + 1}
      ⊢ Eq (HAdd.hAdd Inhabited.default Inhabited.default) Inhabited.default
    -/
    rfl)
    /-
      🎉 no goals
    -/


/-- The free abelian group on a type with one term is isomorphic to `ℤ`. -/
def punitEquiv (T : Type*) [Unique T] : FreeAbelianGroup T ≃+ ℤ where
  toFun := FreeAbelianGroup.lift fun _ ↦ (1 : ℤ)
  invFun n := n • of Inhabited.default
  left_inv z := FreeAbelianGroup.induction_on z
        /-
          α : Type u
          β : Type v
          γ : Type w
          T : Type u_1
          inst✝ : Unique T
          z : FreeAbelianGroup T
          ⊢ Eq ((fun n => HSMul.hSMul n (FreeAbelianGroup.of Inhabited.default)) ((FreeA …
        -/
    (by simp only [zero_smul, AddMonoidHom.map_zero])
        /-
          🎉 no goals
        -/
                               /-
                                 α : Type u
                                 β : Type v
                                 γ : Type w
                                 T : Type u_1
                                 inst✝ : Unique T
                                 z : FreeAbelianGroup T
                                 ⊢ Eq ((fun n => HSMul.hSMul n (FreeAbelianGroup.of Inhabited.default)) ((FreeA …
                               -/
                               /-
                                 🎉 no goals
                               -/
    (Unique.forall_iff.2 <| by simp only [one_smul, lift.of]) (Unique.forall_iff.2 <| by simp)
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    fun x y hx hy ↦ by
      /-
        α : Type u
        β : Type v
        γ : Type w
        T : Type u_1
        inst✝ : Unique T
        z x y : FreeAbelianGroup T
        hx : Eq ((fun n => HSMul.hSMul n (FreeAbelianGroup.of Inhabited.default)) ((Fr …
        hy : Eq ((fun n => HSMul.hSMul n (FreeAbelianGroup.of Inhabited.default)) ((Fr …
        ⊢ Eq ((fun n => HSMul.hSMul n (FreeAbelianGroup.of Inhabited.default)) ((FreeA …
      -/
      simp only [AddMonoidHom.map_add, add_smul] at *
      /-
        α : Type u
        β : Type v
        γ : Type w
        T : Type u_1
        inst✝ : Unique T
        z x y : FreeAbelianGroup T
        hx : Eq (HSMul.hSMul ((FreeAbelianGroup.lift fun x => 1) x) (FreeAbelianGroup. …
        hy : Eq (HSMul.hSMul ((FreeAbelianGroup.lift fun x => 1) y) (FreeAbelianGroup. …
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul ((FreeAbelianGroup.lift fun x => 1) x) (FreeAbeli …
      -/
      rw [hx, hy]
      /-
        🎉 no goals
      -/
  right_inv n := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      T : Type u_1
      inst✝ : Unique T
      n : Int
      ⊢ Eq ((FreeAbelianGroup.lift fun x => 1) ((fun n => HSMul.hSMul n (FreeAbelian …
    -/
    rw [AddMonoidHom.map_zsmul, lift.of]
    /-
      α : Type u
      β : Type v
      γ : Type w
      T : Type u_1
      inst✝ : Unique T
      n : Int
      ⊢ Eq (HSMul.hSMul n 1) n
    -/
    exact zsmul_int_one n
    /-
      🎉 no goals
    -/
  map_add' := AddMonoidHom.map_add _


/-- Isomorphic types have isomorphic free abelian groups. -/
def equivOfEquiv {α β : Type*} (f : α ≃ β) : FreeAbelianGroup α ≃+ FreeAbelianGroup β where
  toFun := map f
  invFun := map f.symm
  left_inv := by
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      ⊢ Function.LeftInverse ⇑(FreeAbelianGroup.map ⇑f.symm) ⇑(FreeAbelianGroup.map  …
    -/
    intro x
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      x : FreeAbelianGroup α
      ⊢ Eq ((FreeAbelianGroup.map ⇑f.symm) ((FreeAbelianGroup.map ⇑f) x)) x
    -/
    rw [← map_comp_apply, Equiv.symm_comp_self, map_id]
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      x : FreeAbelianGroup α
      ⊢ Eq ((AddMonoidHom.id (FreeAbelianGroup α)) x) x
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      ⊢ Function.RightInverse ⇑(FreeAbelianGroup.map ⇑f.symm) ⇑(FreeAbelianGroup.map …
    -/
    intro x
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      x : FreeAbelianGroup β
      ⊢ Eq ((FreeAbelianGroup.map ⇑f) ((FreeAbelianGroup.map ⇑f.symm) x)) x
    -/
    rw [← map_comp_apply, Equiv.self_comp_symm, map_id]
    /-
      α✝ : Type u
      β✝ : Type v
      γ : Type w
      α : Type u_1
      β : Type u_2
      f : Equiv α β
      x : FreeAbelianGroup β
      ⊢ Eq ((AddMonoidHom.id (FreeAbelianGroup β)) x) x
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_add' := AddMonoidHom.map_add _


