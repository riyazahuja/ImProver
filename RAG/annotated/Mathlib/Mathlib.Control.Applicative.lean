theorem Applicative.map_seq_map (f : α → β → γ) (g : σ → β) (x : F α) (y : F σ) :
    f <$> x <*> g <$> y = ((· ∘ g) ∘ f) <$> x <*> y := by
  /-
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ σ : Type u
    f : α → β → γ
    g : σ → β
    x : F α
    y : F σ
    ⊢ Eq (Seq.seq (Functor.map f x) fun x => Functor.map g y) (Seq.seq (Functor.ma …
  -/
  simp [flip, functor_norm, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem Applicative.pure_seq_eq_map' (f : α → β) : ((pure f : F (α → β)) <*> ·) = (f <$> ·) := by
  /-
    F : Type u → Type v
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β : Type u
    f : α → β
    ⊢ Eq (fun x => Seq.seq (Pure.pure f) fun x_1 => x) fun x => Functor.map f x
  -/
  ext; simp [functor_norm]
       /-
         🎉 no goals
       -/


theorem Applicative.ext {F} :
    ∀ {A1 : Applicative F} {A2 : Applicative F} [@LawfulApplicative F A1] [@LawfulApplicative F A2],
      (∀ {α : Type u} (x : α), @Pure.pure _ A1.toPure _ x = @Pure.pure _ A2.toPure _ x) →
      (∀ {α β : Type u} (f : F (α → β)) (x : F α),
          @Seq.seq _ A1.toSeq _ _ f (fun _ => x) = @Seq.seq _ A2.toSeq _ _ f (fun _ => x)) →
      A1 = A2
  | { toFunctor := F1, seq := s1, pure := p1, seqLeft := sl1, seqRight := sr1 },
    { toFunctor := F2, seq := s2, pure := p2, seqLeft := sl2, seqRight := sr2 },
    L1, L2, H1, H2 => by
    obtain rfl : @p1 = @p2 := by
      funext α x
      apply H1
    obtain rfl : @s1 = @s2 := by
      funext α β f x
      exact H2 f (x Unit.unit)
    /-
      F : Type u → Type u_1
      F1 : Functor F
      p1 : {α : Type u} → α → F α
      s1 : {α β : Type u} → F (α → β) → (Unit → F α) → F β
      sl1 : {α β : Type u} → F α → (Unit → F β) → F α
      sr1 : {α β : Type u} → F α → (Unit → F β) → F β
      F2 : Functor F
      sl2 : {α β : Type u} → F α → (Unit → F β) → F α
      sr2 : {α β : Type u} → F α → (Unit → F β) → F β
      L1 : LawfulApplicative F
      L2 : LawfulApplicative F
      H1 : ∀ {α : Type u} (x : α), Eq (Pure.pure x) (Pure.pure x)
      H2 : ∀ {α β : Type u} (f : F (α → β)) (x : F α), Eq (Seq.seq f fun x_1 => x) ( …
      ⊢ Eq Applicative.mk Applicative.mk
    -/
    obtain ⟨seqLeft_eq1, seqRight_eq1, pure_seq1, -⟩ := L1
    /-
      case mk
      F : Type u → Type u_1
      F1 : Functor F
      p1 : {α : Type u} → α → F α
      s1 : {α β : Type u} → F (α → β) → (Unit → F α) → F β
      sl1 : {α β : Type u} → F α → (Unit → F β) → F α
      sr1 : {α β : Type u} → F α → (Unit → F β) → F β
      F2 : Functor F
      sl2 : {α β : Type u} → F α → (Unit → F β) → F α
      sr2 : {α β : Type u} → F α → (Unit → F β) → F β
      L2 : LawfulApplicative F
      H1 : ∀ {α : Type u} (x : α), Eq (Pure.pure x) (Pure.pure x)
      H2 : ∀ {α β : Type u} (f : F (α → β)) (x : F α), Eq (Seq.seq f fun x_1 => x) ( …
      toLawfulFunctor✝ : LawfulFunctor F
      seqLeft_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
      seqRight_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
      pure_seq1 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
      seq_pure✝ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 => …
      seq_assoc✝ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq  …
      ⊢ Eq Applicative.mk Applicative.mk
    -/
    obtain ⟨seqLeft_eq2, seqRight_eq2, pure_seq2, -⟩ := L2
    obtain rfl : F1 = F2 := by
      apply Functor.ext
      intros
      exact (pure_seq1 _ _).symm.trans (pure_seq2 _ _)
    /-
      case mk.mk
      F : Type u → Type u_1
      F1 : Functor F
      p1 : {α : Type u} → α → F α
      s1 : {α β : Type u} → F (α → β) → (Unit → F α) → F β
      sl1 : {α β : Type u} → F α → (Unit → F β) → F α
      sr1 : {α β : Type u} → F α → (Unit → F β) → F β
      sl2 : {α β : Type u} → F α → (Unit → F β) → F α
      sr2 : {α β : Type u} → F α → (Unit → F β) → F β
      toLawfulFunctor✝¹ : LawfulFunctor F
      seqLeft_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
      seqRight_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
      pure_seq1 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
      seq_pure✝¹ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 = …
      seq_assoc✝¹ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq …
      H1 : ∀ {α : Type u} (x : α), Eq (Pure.pure x) (Pure.pure x)
      H2 : ∀ {α β : Type u} (f : F (α → β)) (x : F α), Eq (Seq.seq f fun x_1 => x) ( …
      toLawfulFunctor✝ : LawfulFunctor F
      seqLeft_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
      seqRight_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
      pure_seq2 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
      seq_pure✝ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 => …
      seq_assoc✝ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq  …
      ⊢ Eq Applicative.mk Applicative.mk
    -/
    congr <;> funext α β x y
      /-
        case mk.mk.e_toSeqLeft.e_seqLeft.h.h.h.h
        F : Type u → Type u_1
        F1 : Functor F
        p1 : {α : Type u} → α → F α
        s1 : {α β : Type u} → F (α → β) → (Unit → F α) → F β
        sl1 : {α β : Type u} → F α → (Unit → F β) → F α
        sr1 : {α β : Type u} → F α → (Unit → F β) → F β
        sl2 : {α β : Type u} → F α → (Unit → F β) → F α
        sr2 : {α β : Type u} → F α → (Unit → F β) → F β
        toLawfulFunctor✝¹ : LawfulFunctor F
        seqLeft_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
        seqRight_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
        pure_seq1 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
        seq_pure✝¹ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 = …
        seq_assoc✝¹ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq …
        H1 : ∀ {α : Type u} (x : α), Eq (Pure.pure x) (Pure.pure x)
        H2 : ∀ {α β : Type u} (f : F (α → β)) (x : F α), Eq (Seq.seq f fun x_1 => x) ( …
        toLawfulFunctor✝ : LawfulFunctor F
        seqLeft_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
        seqRight_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
        pure_seq2 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
        seq_pure✝ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 => …
        seq_assoc✝ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq  …
        α β : Type u
        x : F α
        y : Unit → F β
        ⊢ Eq (sl1 x y) (sl2 x y)
      -/
    · exact (seqLeft_eq1 _ (y Unit.unit)).trans (seqLeft_eq2 _ _).symm
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.e_toSeqRight.e_seqRight.h.h.h.h
        F : Type u → Type u_1
        F1 : Functor F
        p1 : {α : Type u} → α → F α
        s1 : {α β : Type u} → F (α → β) → (Unit → F α) → F β
        sl1 : {α β : Type u} → F α → (Unit → F β) → F α
        sr1 : {α β : Type u} → F α → (Unit → F β) → F β
        sl2 : {α β : Type u} → F α → (Unit → F β) → F α
        sr2 : {α β : Type u} → F α → (Unit → F β) → F β
        toLawfulFunctor✝¹ : LawfulFunctor F
        seqLeft_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
        seqRight_eq1 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
        pure_seq1 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
        seq_pure✝¹ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 = …
        seq_assoc✝¹ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq …
        H1 : ∀ {α : Type u} (x : α), Eq (Pure.pure x) (Pure.pure x)
        H2 : ∀ {α β : Type u} (f : F (α → β)) (x : F α), Eq (Seq.seq f fun x_1 => x) ( …
        toLawfulFunctor✝ : LawfulFunctor F
        seqLeft_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqLeft.seqLeft x fun  …
        seqRight_eq2 : ∀ {α β : Type u} (x : F α) (y : F β), Eq (SeqRight.seqRight x f …
        pure_seq2 : ∀ {α β : Type u} (g : α → β) (x : F α), Eq (Seq.seq (Pure.pure g)  …
        seq_pure✝ : ∀ {α β : Type u} (g : F (α → β)) (x : α), Eq (Seq.seq g fun x_1 => …
        seq_assoc✝ : ∀ {α β γ : Type u} (x : F α) (g : F (α → β)) (h : F (β → γ)), Eq  …
        α β : Type u
        x : F α
        y : Unit → F β
        ⊢ Eq (sr1 x y) (sr2 x y)
      -/
    · exact (seqRight_eq1 _ (y Unit.unit)).trans (seqRight_eq2 _ (y Unit.unit)).symm
      /-
        🎉 no goals
      -/


instance : CommApplicative Id where commutative_prod _ _ := rfl


theorem map_pure (f : α → β) (x : α) : (f <$> pure x : Comp F G β) = pure (f x) :=
                 /-
                   F : Type u → Type w
                   G : Type v → Type u
                   inst✝³ : Applicative F
                   inst✝² : Applicative G
                   inst✝¹ : LawfulApplicative F
                   inst✝ : LawfulApplicative G
                   α β : Type v
                   f : α → β
                   x : α
                   ⊢ Eq (Functor.map f (Pure.pure x)).run (Pure.pure (f x)).run
                 -/
  Comp.ext <| by simp
                 /-
                   🎉 no goals
                 -/


theorem seq_pure (f : Comp F G (α → β)) (x : α) : f <*> pure x = (fun g : α → β => g x) <$> f :=
                 /-
                   F : Type u → Type w
                   G : Type v → Type u
                   inst✝³ : Applicative F
                   inst✝² : Applicative G
                   inst✝¹ : LawfulApplicative F
                   inst✝ : LawfulApplicative G
                   α β : Type v
                   f : Functor.Comp F G (α → β)
                   x : α
                   ⊢ Eq (Seq.seq f fun x_1 => Pure.pure x).run (Functor.map (fun g => g x) f).run
                 -/
  Comp.ext <| by simp [comp_def, functor_norm]
                 /-
                   🎉 no goals
                 -/


theorem seq_assoc (x : Comp F G α) (f : Comp F G (α → β)) (g : Comp F G (β → γ)) :
    g <*> (f <*> x) = @Function.comp α β γ <$> g <*> f <*> x :=
                 /-
                   F : Type u → Type w
                   G : Type v → Type u
                   inst✝³ : Applicative F
                   inst✝² : Applicative G
                   inst✝¹ : LawfulApplicative F
                   inst✝ : LawfulApplicative G
                   α β γ : Type v
                   x : Functor.Comp F G α
                   f : Functor.Comp F G (α → β)
                   g : Functor.Comp F G (β → γ)
                   ⊢ Eq (Seq.seq g fun x_1 => Seq.seq f fun x_2 => x).run (Seq.seq (Seq.seq (Func …
                 -/
  Comp.ext <| by simp [comp_def, functor_norm]
                 /-
                   🎉 no goals
                 -/


theorem pure_seq_eq_map (f : α → β) (x : Comp F G α) : pure f <*> x = f <$> x :=
                 /-
                   F : Type u → Type w
                   G : Type v → Type u
                   inst✝³ : Applicative F
                   inst✝² : Applicative G
                   inst✝¹ : LawfulApplicative F
                   inst✝ : LawfulApplicative G
                   α β : Type v
                   f : α → β
                   x : Functor.Comp F G α
                   ⊢ Eq (Seq.seq (Pure.pure f) fun x_1 => x).run (Functor.map f x).run
                 -/
  Comp.ext <| by simp [Applicative.pure_seq_eq_map', functor_norm]
                 /-
                   🎉 no goals
                 -/

-- TODO: the first two results were handled by `control_laws_tac` in mathlib3

instance instLawfulApplicativeComp : LawfulApplicative (Comp F G) where
                   /-
                     F : Type u → Type w
                     G : Type v → Type u
                     inst✝³ : Applicative F
                     inst✝² : Applicative G
                     inst✝¹ : LawfulApplicative F
                     inst✝ : LawfulApplicative G
                     α β γ : Type v
                     ⊢ ∀ {α β : Type v} (x : Functor.Comp F G α) (y : Functor.Comp F G β), Eq (SeqL …
                   -/
  seqLeft_eq := by intros; rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      F : Type u → Type w
                      G : Type v → Type u
                      inst✝³ : Applicative F
                      inst✝² : Applicative G
                      inst✝¹ : LawfulApplicative F
                      inst✝ : LawfulApplicative G
                      α β γ : Type v
                      ⊢ ∀ {α β : Type v} (x : Functor.Comp F G α) (y : Functor.Comp F G β), Eq (SeqR …
                    -/
  seqRight_eq := by intros; rfl
                            /-
                              🎉 no goals
                            -/
  pure_seq := Comp.pure_seq_eq_map
  map_pure := Comp.map_pure
  seq_pure := Comp.seq_pure
  seq_assoc := Comp.seq_assoc

-- Porting note: mathport wasn't aware of the new implicit parameter omission in these `fun` binders


theorem applicative_id_comp {F} [AF : Applicative F] [LawfulApplicative F] :
    @instApplicativeComp Id F _ _ = AF :=
  @Applicative.ext F _ _ (instLawfulApplicativeComp (F := Id)) _
    (fun _ => rfl) (fun _ _ => rfl)


theorem applicative_comp_id {F} [AF : Applicative F] [LawfulApplicative F] :
    @Comp.instApplicativeComp F Id _ _ = AF :=
  @Applicative.ext F _ _ (instLawfulApplicativeComp (G := Id)) _
                                                                /-
                                                                  F : Type u_1 → Type u_2
                                                                  AF : Applicative F
                                                                  inst✝ : LawfulApplicative F
                                                                  α✝ β✝ : Type u_1
                                                                  f : F (α✝ → β✝)
                                                                  x : F α✝
                                                                  ⊢ Eq (Seq.seq (Functor.map id f) fun x_1 => x) (Seq.seq f fun x_1 => x)
                                                                -/
    (fun _ => rfl) (fun f x => show id <$> f <*> x = f <*> x by rw [id_map])
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance {f : Type u → Type w} {g : Type v → Type u} [Applicative f] [Applicative g]
    [CommApplicative f] [CommApplicative g] : CommApplicative (Comp f g) where
  commutative_prod _ _ := by
    /-
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      ⊢ Eq (Seq.seq (Functor.map Prod.mk x✝¹) fun x => x✝) (Seq.seq (Functor.map (fu …
    -/
    simp! [map, Seq.seq]
    /-
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      ⊢ Eq (Functor.Comp.mk (Seq.seq (Functor.map (fun x1 x2 => Seq.seq x1 fun x =>  …
    -/
    rw [commutative_map]
    /-
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      ⊢ Eq (Functor.Comp.mk (Seq.seq (Functor.map (flip fun x1 x2 => Seq.seq x1 fun  …
    -/
    simp only [mk, flip, seq_map_assoc, Function.comp_def, map_map]
    /-
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      ⊢ Eq (Seq.seq (Functor.map (fun a x => Seq.seq (Functor.map Prod.mk x) fun x = …
    -/
    congr
    /-
      case e_a.e_a
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      ⊢ Eq (fun a x => Seq.seq (Functor.map Prod.mk x) fun x => a) fun a x2 => Seq.s …
    -/
    funext x y
    /-
      case e_a.e_a.h.h
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      x : g β✝
      y : g α✝
      ⊢ Eq (Seq.seq (Functor.map Prod.mk y) fun x_1 => x) (Seq.seq (Functor.map (fun …
    -/
    rw [commutative_map]
    /-
      case e_a.e_a.h.h
      F : Type u → Type w
      G : Type v → Type u
      inst✝⁷ : Applicative F
      inst✝⁶ : Applicative G
      inst✝⁵ : LawfulApplicative F
      inst✝⁴ : LawfulApplicative G
      α β γ : Type v
      f : Type u → Type w
      g : Type v → Type u
      inst✝³ : Applicative f
      inst✝² : Applicative g
      inst✝¹ : CommApplicative f
      inst✝ : CommApplicative g
      α✝ β✝ : Type v
      x✝¹ : Functor.Comp f g α✝
      x✝ : Functor.Comp f g β✝
      x : g β✝
      y : g α✝
      ⊢ Eq (Seq.seq (Functor.map (flip Prod.mk) x) fun x => y) (Seq.seq (Functor.map …
    -/
    congr
    /-
      🎉 no goals
    -/


@[functor_norm]
theorem Comp.seq_mk {α β : Type w} {f : Type u → Type v} {g : Type w → Type u} [Applicative f]
    [Applicative g] (h : f (g (α → β))) (x : f (g α)) :
    Comp.mk h <*> Comp.mk x = Comp.mk ((· <*> ·) <$> h <*> x) :=
  rfl

-- Porting note: There is some awkwardness in the following definition now that we have `HMul`.


instance {α} [One α] [Mul α] : Applicative (Const α) where
  pure _ := (1 : α)
  seq f x := (show α from f) * (show α from x Unit.unit)

-- Porting note: `(· <*> ·)` needed to change to `Seq.seq` in the `simp`.
-- Also, `simp` didn't close `refl` goals.


instance {α} [Monoid α] : LawfulApplicative (Const α) where
  map_pure _ _ := rfl
                     /-
                       α : Type u_1
                       inst✝ : Monoid α
                       α✝ β✝ : Type u_2
                       x✝¹ : Functor.Const α (α✝ → β✝)
                       x✝ : α✝
                       ⊢ Eq (Seq.seq x✝¹ fun x => Pure.pure x✝) (Functor.map (fun h => h x✝) x✝¹)
                     -/
                     /-
                       α : Type u_1
                       inst✝ : Monoid α
                       α✝ β✝ : Type u_2
                       x✝¹ : α✝ → β✝
                       x✝ : Functor.Const α α✝
                       ⊢ Eq (Seq.seq (Pure.pure x✝¹) fun x => x✝) (Functor.map x✝¹ x✝)
                     -/
                       /-
                         α : Type u_1
                         inst✝ : Monoid α
                         α✝ β✝ : Type u_2
                         x✝¹ : Functor.Const α α✝
                         x✝ : Functor.Const α β✝
                         ⊢ Eq (SeqLeft.seqLeft x✝¹ fun x => x✝) (Seq.seq (Functor.map (Function.const β …
                       -/
  seq_pure _ _ := by simp only [Seq.seq, pure, mul_one]; rfl
                                            /-
                                              🎉 no goals
                                            -/
                        /-
                          α : Type u_1
                          inst✝ : Monoid α
                          α✝ β✝ : Type u_2
                          x✝¹ : Functor.Const α α✝
                          x✝ : Functor.Const α β✝
                          ⊢ Eq (SeqRight.seqRight x✝¹ fun x => x✝) (Seq.seq (Functor.map (Function.const …
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
  pure_seq _ _ := by simp only [Seq.seq, pure, one_mul]; rfl
  seqLeft_eq _ _ := by simp only [Seq.seq]; rfl
  seqRight_eq _ _ := by simp only [Seq.seq]; rfl
                        /-
                          α : Type u_1
                          inst✝ : Monoid α
                          α✝ β✝ γ✝ : Type u_2
                          x✝² : Functor.Const α α✝
                          x✝¹ : Functor.Const α (α✝ → β✝)
                          x✝ : Functor.Const α (β✝ → γ✝)
                          ⊢ Eq (Seq.seq x✝ fun x => Seq.seq x✝¹ fun x => x✝²) (Seq.seq (Seq.seq (Functor …
                        -/
  seq_assoc _ _ _ := by simp only [Seq.seq, mul_assoc]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


instance {α} [Zero α] [Add α] : Applicative (AddConst α) where
  pure _ := (0 : α)
  seq f x := (show α from f) + (show α from x Unit.unit)


instance {α} [AddMonoid α] : LawfulApplicative (AddConst α) where
  map_pure _ _ := rfl
                     /-
                       α : Type u_1
                       inst✝ : AddMonoid α
                       α✝ β✝ : Type u_2
                       x✝¹ : Functor.AddConst α (α✝ → β✝)
                       x✝ : α✝
                       ⊢ Eq (Seq.seq x✝¹ fun x => Pure.pure x✝) (Functor.map (fun h => h x✝) x✝¹)
                     -/
                     /-
                       α : Type u_1
                       inst✝ : AddMonoid α
                       α✝ β✝ : Type u_2
                       x✝¹ : α✝ → β✝
                       x✝ : Functor.AddConst α α✝
                       ⊢ Eq (Seq.seq (Pure.pure x✝¹) fun x => x✝) (Functor.map x✝¹ x✝)
                     -/
                       /-
                         α : Type u_1
                         inst✝ : AddMonoid α
                         α✝ β✝ : Type u_2
                         x✝¹ : Functor.AddConst α α✝
                         x✝ : Functor.AddConst α β✝
                         ⊢ Eq (SeqLeft.seqLeft x✝¹ fun x => x✝) (Seq.seq (Functor.map (Function.const β …
                       -/
  seq_pure _ _ := by simp only [Seq.seq, pure, add_zero]; rfl
                                            /-
                                              🎉 no goals
                                            -/
                        /-
                          α : Type u_1
                          inst✝ : AddMonoid α
                          α✝ β✝ : Type u_2
                          x✝¹ : Functor.AddConst α α✝
                          x✝ : Functor.AddConst α β✝
                          ⊢ Eq (SeqRight.seqRight x✝¹ fun x => x✝) (Seq.seq (Functor.map (Function.const …
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
  pure_seq _ _ := by simp only [Seq.seq, pure, zero_add]; rfl
  seqLeft_eq _ _ := by simp only [Seq.seq]; rfl
  seqRight_eq _ _ := by simp only [Seq.seq]; rfl
                        /-
                          α : Type u_1
                          inst✝ : AddMonoid α
                          α✝ β✝ γ✝ : Type u_2
                          x✝² : Functor.AddConst α α✝
                          x✝¹ : Functor.AddConst α (α✝ → β✝)
                          x✝ : Functor.AddConst α (β✝ → γ✝)
                          ⊢ Eq (Seq.seq x✝ fun x => Seq.seq x✝¹ fun x => x✝²) (Seq.seq (Seq.seq (Functor …
                        -/
  seq_assoc _ _ _ := by simp only [Seq.seq, add_assoc]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/

