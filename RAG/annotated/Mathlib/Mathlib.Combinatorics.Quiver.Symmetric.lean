/-- A type synonym for the symmetrized quiver (with an arrow both ways for each original arrow).
    NB: this does not work for `Prop`-valued quivers. It requires `[Quiver.{v+1} V]`. -/
-- Porting note: no hasNonemptyInstance linter yet
def Symmetrify (V : Type*) := V


instance symmetrifyQuiver (V : Type u) [Quiver V] : Quiver (Symmetrify V) :=
  ⟨fun a b : V ↦ (a ⟶ b) ⊕ (b ⟶ a)⟩


/-- A quiver `HasReverse` if we can reverse an arrow `p` from `a` to `b` to get an arrow
    `p.reverse` from `b` to `a`. -/
class HasReverse where
  /-- the map which sends an arrow to its reverse -/
  reverse' : ∀ {a b : V}, (a ⟶ b) → (b ⟶ a)


/-- Reverse the direction of an arrow. -/
def reverse {V} [Quiver.{v + 1} V] [HasReverse V] {a b : V} : (a ⟶ b) → (b ⟶ a) :=
  HasReverse.reverse'


/-- A quiver `HasInvolutiveReverse` if reversing twice is the identity. -/
class HasInvolutiveReverse extends HasReverse V where
  /-- `reverse` is involutive -/
  inv' : ∀ {a b : V} (f : a ⟶ b), reverse (reverse f) = f


@[simp]
theorem reverse_reverse [h : HasInvolutiveReverse V] {a b : V} (f : a ⟶ b) :
                                  /-
                                    V : Type u_2
                                    inst✝ : Quiver V
                                    h : Quiver.HasInvolutiveReverse V
                                    a b : V
                                    f : Quiver.Hom a b
                                    ⊢ Eq (Quiver.reverse (Quiver.reverse f)) f
                                  -/
    reverse (reverse f) = f := by apply h.inv'
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem reverse_inj [h : HasInvolutiveReverse V] {a b : V}
    (f g : a ⟶ b) : reverse f = reverse g ↔ f = g := by
  /-
    V : Type u_2
    inst✝ : Quiver V
    h : Quiver.HasInvolutiveReverse V
    a b : V
    f g : Quiver.Hom a b
    ⊢ Iff (Eq (Quiver.reverse f) (Quiver.reverse g)) (Eq f g)
  -/
  constructor
    /-
      case mp
      V : Type u_2
      inst✝ : Quiver V
      h : Quiver.HasInvolutiveReverse V
      a b : V
      f g : Quiver.Hom a b
      ⊢ Eq (Quiver.reverse f) (Quiver.reverse g) → Eq f g
    -/
  · rintro h
    /-
      case mp
      V : Type u_2
      inst✝ : Quiver V
      h✝ : Quiver.HasInvolutiveReverse V
      a b : V
      f g : Quiver.Hom a b
      h : Eq (Quiver.reverse f) (Quiver.reverse g)
      ⊢ Eq f g
    -/
    simpa using congr_arg Quiver.reverse h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_2
      inst✝ : Quiver V
      h : Quiver.HasInvolutiveReverse V
      a b : V
      f g : Quiver.Hom a b
      ⊢ Eq f g → Eq (Quiver.reverse f) (Quiver.reverse g)
    -/
  · rintro h
    /-
      case mpr
      V : Type u_2
      inst✝ : Quiver V
      h✝ : Quiver.HasInvolutiveReverse V
      a b : V
      f g : Quiver.Hom a b
      h : Eq f g
      ⊢ Eq (Quiver.reverse f) (Quiver.reverse g)
    -/
    congr
    /-
      🎉 no goals
    -/


theorem eq_reverse_iff [h : HasInvolutiveReverse V] {a b : V} (f : a ⟶ b)
    (g : b ⟶ a) : f = reverse g ↔ reverse f = g := by
  /-
    V : Type u_2
    inst✝ : Quiver V
    h : Quiver.HasInvolutiveReverse V
    a b : V
    f : Quiver.Hom a b
    g : Quiver.Hom b a
    ⊢ Iff (Eq f (Quiver.reverse g)) (Eq (Quiver.reverse f) g)
  -/
  rw [← reverse_inj, reverse_reverse]
  /-
    🎉 no goals
  -/


/-- A prefunctor preserving reversal of arrows -/
class _root_.Prefunctor.MapReverse (φ : U ⥤q V) : Prop where
  /-- The image of a reverse is the reverse of the image. -/
  map_reverse' : ∀ {u v : U} (e : u ⟶ v), φ.map (reverse e) = reverse (φ.map e)


@[simp]
theorem _root_.Prefunctor.map_reverse (φ : U ⥤q V) [φ.MapReverse]
    {u v : U} (e : u ⟶ v) : φ.map (reverse e) = reverse (φ.map e) :=
  Prefunctor.MapReverse.map_reverse' e


instance _root_.Prefunctor.mapReverseComp
    (φ : U ⥤q V) (ψ : V ⥤q W) [φ.MapReverse] [ψ.MapReverse] :
    (φ ⋙q ψ).MapReverse where
  map_reverse' e := by
    /-
      U : Type u_1
      V : Type u_2
      W : Type u_3
      inst✝⁷ : Quiver U
      inst✝⁶ : Quiver V
      inst✝⁵ : Quiver W
      inst✝⁴ : Quiver.HasReverse U
      inst✝³ : Quiver.HasReverse V
      inst✝² : Quiver.HasReverse W
      φ : Prefunctor U V
      ψ : Prefunctor V W
      inst✝¹ : φ.MapReverse
      inst✝ : ψ.MapReverse
      u✝ v✝ : U
      e : Quiver.Hom u✝ v✝
      ⊢ Eq ((φ.comp ψ).map (Quiver.reverse e)) (Quiver.reverse ((φ.comp ψ).map e))
    -/
    simp only [Prefunctor.comp_map, Prefunctor.MapReverse.map_reverse']
    /-
      🎉 no goals
    -/


instance _root_.Prefunctor.mapReverseId :
    (Prefunctor.id U).MapReverse where
  map_reverse' _ := rfl


instance : HasReverse (Symmetrify V) :=
  ⟨fun e => e.swap⟩


instance :
    HasInvolutiveReverse
      (Symmetrify V) where
  toHasReverse := ⟨fun e ↦ e.swap⟩
  inv' e := congr_fun Sum.swap_swap_eq e


@[simp]
theorem symmetrify_reverse {a b : Symmetrify V} (e : a ⟶ b) : reverse e = e.swap :=
  rfl


/-- Shorthand for the "forward" arrow corresponding to `f` in `symmetrify V` -/
abbrev Hom.toPos {X Y : V} (f : X ⟶ Y) : (Quiver.symmetrifyQuiver V).Hom X Y :=
  Sum.inl f


/-- Shorthand for the "backward" arrow corresponding to `f` in `symmetrify V` -/
abbrev Hom.toNeg {X Y : V} (f : X ⟶ Y) : (Quiver.symmetrifyQuiver V).Hom Y X :=
  Sum.inr f


/-- Reverse the direction of a path. -/
@[simp]
def Path.reverse [HasReverse V] {a : V} : ∀ {b}, Path a b → Path b a
  | _, Path.nil => Path.nil
  | _, Path.cons p e => (Quiver.reverse e).toPath.comp p.reverse


@[simp]
theorem Path.reverse_toPath [HasReverse V] {a b : V} (f : a ⟶ b) :
    f.toPath.reverse = (Quiver.reverse f).toPath :=
  rfl


@[simp]
theorem Path.reverse_comp [HasReverse V] {a b c : V} (p : Path a b) (q : Path b c) :
    (p.comp q).reverse = q.reverse.comp p.reverse := by
  induction q with
  | nil => simp
  | cons _ _ h => simp [h]


@[simp]
theorem Path.reverse_reverse [h : HasInvolutiveReverse V] {a b : V} (p : Path a b) :
    p.reverse.reverse = p := by
  induction p with
  | nil => simp
  | cons _ _ h =>
    rw [Path.reverse, Path.reverse_comp, h, Path.reverse_toPath, Quiver.reverse_reverse]
    rfl


/-- The inclusion of a quiver in its symmetrification -/
def of : Prefunctor V (Symmetrify V) where
  obj := id
  map := Sum.inl


/-- Given a quiver `V'` with reversible arrows, a prefunctor to `V'` can be lifted to one from
    `Symmetrify V` to `V'` -/
def lift [HasReverse V'] (φ : Prefunctor V V') :
    Prefunctor (Symmetrify V) V' where
  obj := φ.obj
  map f := match f with
  | Sum.inl g => φ.map g
  | Sum.inr g => reverse (φ.map g)


theorem lift_spec [HasReverse V'] (φ : Prefunctor V V') :
    Symmetrify.of.comp (Symmetrify.lift φ) = φ := by
  /-
    V : Type u_2
    inst✝² : Quiver V
    V' : Type u_4
    inst✝¹ : Quiver V'
    inst✝ : Quiver.HasReverse V'
    φ : Prefunctor V V'
    ⊢ Eq (Quiver.Symmetrify.of.comp (Quiver.Symmetrify.lift φ)) φ
  -/
  fapply Prefunctor.ext
    /-
      case h_obj
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      φ : Prefunctor V V'
      ⊢ ∀ (X : V), Eq ((Quiver.Symmetrify.of.comp (Quiver.Symmetrify.lift φ)).obj X) …
    -/
  · rintro X
    /-
      case h_obj
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      φ : Prefunctor V V'
      X : V
      ⊢ Eq ((Quiver.Symmetrify.of.comp (Quiver.Symmetrify.lift φ)).obj X) (φ.obj X)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      φ : Prefunctor V V'
      ⊢ ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ((Quiver.Symmetrify.of.comp (Quiver.Sym …
    -/
  · rintro X Y f
    /-
      case h_map
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      φ : Prefunctor V V'
      X Y : V
      f : Quiver.Hom X Y
      ⊢ Eq ((Quiver.Symmetrify.of.comp (Quiver.Symmetrify.lift φ)).map f) (Eq.recOn  …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem lift_reverse [h : HasInvolutiveReverse V']
    (φ : Prefunctor V V') {X Y : Symmetrify V} (f : X ⟶ Y) :
    (Symmetrify.lift φ).map (Quiver.reverse f) = Quiver.reverse ((Symmetrify.lift φ).map f) := by
  /-
    V : Type u_2
    inst✝¹ : Quiver V
    V' : Type u_4
    inst✝ : Quiver V'
    h : Quiver.HasInvolutiveReverse V'
    φ : Prefunctor V V'
    X Y : Quiver.Symmetrify V
    f : Quiver.Hom X Y
    ⊢ Eq ((Quiver.Symmetrify.lift φ).map (Quiver.reverse f)) (Quiver.reverse ((Qui …
  -/
  dsimp [Symmetrify.lift]; cases f
    /-
      case inl
      V : Type u_2
      inst✝¹ : Quiver V
      V' : Type u_4
      inst✝ : Quiver V'
      h : Quiver.HasInvolutiveReverse V'
      φ : Prefunctor V V'
      X Y : Quiver.Symmetrify V
      val✝ : Quiver.Hom X Y
      ⊢ Eq (Quiver.Symmetrify.lift.match_1 (fun f => Quiver.Hom (φ.obj Y) (φ.obj X)) …
    -/
  · simp only
    /-
      case inl
      V : Type u_2
      inst✝¹ : Quiver V
      V' : Type u_4
      inst✝ : Quiver V'
      h : Quiver.HasInvolutiveReverse V'
      φ : Prefunctor V V'
      X Y : Quiver.Symmetrify V
      val✝ : Quiver.Hom X Y
      ⊢ Eq (Quiver.Symmetrify.lift.match_1 (fun f => Quiver.Hom (φ.obj Y) (φ.obj X)) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_2
      inst✝¹ : Quiver V
      V' : Type u_4
      inst✝ : Quiver V'
      h : Quiver.HasInvolutiveReverse V'
      φ : Prefunctor V V'
      X Y : Quiver.Symmetrify V
      val✝ : Quiver.Hom Y X
      ⊢ Eq (Quiver.Symmetrify.lift.match_1 (fun f => Quiver.Hom (φ.obj Y) (φ.obj X)) …
    -/
  · simp only [reverse_reverse]
    /-
      case inr
      V : Type u_2
      inst✝¹ : Quiver V
      V' : Type u_4
      inst✝ : Quiver V'
      h : Quiver.HasInvolutiveReverse V'
      φ : Prefunctor V V'
      X Y : Quiver.Symmetrify V
      val✝ : Quiver.Hom Y X
      ⊢ Eq (Quiver.Symmetrify.lift.match_1 (fun f => Quiver.Hom (φ.obj Y) (φ.obj X)) …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- `lift φ` is the only prefunctor extending `φ` and preserving reverses. -/
theorem lift_unique [HasReverse V'] (φ : V ⥤q V') (Φ : Symmetrify V ⥤q V') (hΦ : (of ⋙q Φ) = φ)
    (hΦinv : ∀ {X Y : Symmetrify V} (f : X ⟶ Y),
      Φ.map (Quiver.reverse f) = Quiver.reverse (Φ.map f)) :
    Φ = Symmetrify.lift φ := by
  /-
    V : Type u_2
    inst✝² : Quiver V
    V' : Type u_4
    inst✝¹ : Quiver V'
    inst✝ : Quiver.HasReverse V'
    φ : Prefunctor V V'
    Φ : Prefunctor (Quiver.Symmetrify V) V'
    hΦ : Eq (Quiver.Symmetrify.of.comp Φ) φ
    hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
    ⊢ Eq Φ (Quiver.Symmetrify.lift φ)
  -/
  subst_vars
  /-
    V : Type u_2
    inst✝² : Quiver V
    V' : Type u_4
    inst✝¹ : Quiver V'
    inst✝ : Quiver.HasReverse V'
    Φ : Prefunctor (Quiver.Symmetrify V) V'
    hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
    ⊢ Eq Φ (Quiver.Symmetrify.lift (Quiver.Symmetrify.of.comp Φ))
  -/
  fapply Prefunctor.ext
    /-
      case h_obj
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      Φ : Prefunctor (Quiver.Symmetrify V) V'
      hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
      ⊢ ∀ (X : Quiver.Symmetrify V), Eq (Φ.obj X) ((Quiver.Symmetrify.lift (Quiver.S …
    -/
  · rintro X
    /-
      case h_obj
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      Φ : Prefunctor (Quiver.Symmetrify V) V'
      hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
      X : Quiver.Symmetrify V
      ⊢ Eq (Φ.obj X) ((Quiver.Symmetrify.lift (Quiver.Symmetrify.of.comp Φ)).obj X)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      Φ : Prefunctor (Quiver.Symmetrify V) V'
      hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
      ⊢ ∀ (X Y : Quiver.Symmetrify V) (f : Quiver.Hom X Y), Eq (Φ.map f) (Eq.recOn ⋯ …
    -/
  · rintro X Y f
    /-
      case h_map
      V : Type u_2
      inst✝² : Quiver V
      V' : Type u_4
      inst✝¹ : Quiver V'
      inst✝ : Quiver.HasReverse V'
      Φ : Prefunctor (Quiver.Symmetrify V) V'
      hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
      X Y : Quiver.Symmetrify V
      f : Quiver.Hom X Y
      ⊢ Eq (Φ.map f) (Eq.recOn ⋯ (Eq.recOn ⋯ ((Quiver.Symmetrify.lift (Quiver.Symmet …
    -/
    cases f
      /-
        case h_map.inl
        V : Type u_2
        inst✝² : Quiver V
        V' : Type u_4
        inst✝¹ : Quiver V'
        inst✝ : Quiver.HasReverse V'
        Φ : Prefunctor (Quiver.Symmetrify V) V'
        hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
        X Y : Quiver.Symmetrify V
        val✝ : Quiver.Hom X Y
        ⊢ Eq (Φ.map (Sum.inl val✝)) (Eq.recOn ⋯ (Eq.recOn ⋯ ((Quiver.Symmetrify.lift ( …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h_map.inr
        V : Type u_2
        inst✝² : Quiver V
        V' : Type u_4
        inst✝¹ : Quiver V'
        inst✝ : Quiver.HasReverse V'
        Φ : Prefunctor (Quiver.Symmetrify V) V'
        hΦinv : ∀ {X Y : Quiver.Symmetrify V} (f : Quiver.Hom X Y), Eq (Φ.map (Quiver. …
        X Y : Quiver.Symmetrify V
        val✝ : Quiver.Hom Y X
        ⊢ Eq (Φ.map (Sum.inr val✝)) (Eq.recOn ⋯ (Eq.recOn ⋯ ((Quiver.Symmetrify.lift ( …
      -/
    · exact hΦinv (Sum.inl _)
      /-
        🎉 no goals
      -/


/-- A prefunctor canonically defines a prefunctor of the symmetrifications. -/
@[simps]
def _root_.Prefunctor.symmetrify (φ : U ⥤q V) : Symmetrify U ⥤q Symmetrify V where
  obj := φ.obj
  map := Sum.map φ.map φ.map


instance _root_.Prefunctor.symmetrify_mapReverse (φ : U ⥤q V) :
    Prefunctor.MapReverse φ.symmetrify :=
               /-
                 U : Type u_1
                 V : Type u_2
                 W : Type u_3
                 inst✝³ : Quiver U
                 inst✝² : Quiver V
                 inst✝¹ : Quiver W
                 V' : Type u_4
                 inst✝ : Quiver V'
                 φ : Prefunctor U V
                 u✝ v✝ : Quiver.Symmetrify U
                 e : Quiver.Hom u✝ v✝
                 ⊢ Eq (φ.symmetrify.map (Quiver.reverse e)) (Quiver.reverse (φ.symmetrify.map e))
               -/
                           /-
                             🎉 no goals
                           -/
  ⟨fun e => by cases e <;> rfl⟩
                           /-
                             🎉 no goals
                           -/


instance [HasReverse V] : HasReverse (Quiver.Push σ) where
  reverse' := fun
              | PushQuiver.arrow f => PushQuiver.arrow (reverse f)


instance [h : HasInvolutiveReverse V] :
    HasInvolutiveReverse (Push σ) where
  reverse' := fun
  | PushQuiver.arrow f => PushQuiver.arrow (reverse f)
  inv' := fun
                             /-
                               U : Type u_1
                               V : Type u_2
                               W : Type u_3
                               inst✝² : Quiver U
                               inst✝¹ : Quiver V
                               inst✝ : Quiver W
                               V' : Type u_4
                               σ : V → V'
                               h : Quiver.HasInvolutiveReverse V
                               a✝ b✝ : Quiver.Push σ
                               X✝ Y✝ : V
                               f : Quiver.Hom X✝ Y✝
                               ⊢ Eq (Quiver.reverse (Quiver.reverse (Quiver.PushQuiver.arrow f))) (Quiver.Pus …
                             -/
  | PushQuiver.arrow f => by dsimp [reverse]; congr; apply h.inv'
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem of_reverse [HasInvolutiveReverse V] (X Y : V) (f : X ⟶ Y) :
    (reverse <| (Push.of σ).map f) = (Push.of σ).map (reverse f) :=
  rfl


instance ofMapReverse [h : HasInvolutiveReverse V] : (Push.of σ).MapReverse :=
      /-
        U : Type u_1
        V : Type u_2
        W : Type u_3
        inst✝² : Quiver U
        inst✝¹ : Quiver V
        inst✝ : Quiver W
        V' : Type u_4
        σ : V → V'
        h : Quiver.HasInvolutiveReverse V
        ⊢ ∀ {u v : V} (e : Quiver.Hom u v), Eq ((Quiver.Push.of σ).map (Quiver.reverse …
      -/
  ⟨by simp [of_reverse]⟩
      /-
        🎉 no goals
      -/


/-- A quiver is preconnected iff there exists a path between any pair of
vertices.
Note that if `V` doesn't `HasReverse`, then the definition is stronger than
simply having a preconnected underlying `SimpleGraph`, since a path in one
direction doesn't induce one in the other.
-/
def IsPreconnected (V) [Quiver.{u + 1} V] :=
  ∀ X Y : V, Nonempty (Path X Y)


