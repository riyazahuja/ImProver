/-- Function that extracts the name of the corresponding `Algebra` property from a `RingHom`
property that has been tagged with the `algebraize` attribute. This is done by either returning the
parameter of the attribute, or by assuming that the tagged declaration has name `RingHom.Property`
and then returning `Algebra.Property`. -/
def algebraizeGetParam (thm : Name) (stx : Syntax) : AttrM Name := do
  match stx with
  | `(attr| algebraize $name:ident) => return name.getId
  /- If no argument is provided, assume `thm` is of the form `RingHom.Property`,
  and return `Algebra.Property` -/
  | `(attr| algebraize) =>
    match thm with
    | .str `RingHom t => return .str `Algebra t
    | _ =>
      throwError "theorem name must be of the form `RingHom.Property` if no argument is provided"
  | _ => throwError "unexpected algebraize argument"


/-- A user attribute that is used to tag `RingHom` properties that can be converted to `Algebra`
properties. Using an (optional) parameter, it will also generate a `Name` of a declaration which
will help the `algebraize` tactic access the corresponding `Algebra` property.

There are two cases for what declaration corresponding to this `Name` can be.

1. An inductive type (i.e. the `Algebra` property itself), in this case it is assumed that the
`RingHom` and the `Algebra` property are definitionally the same, and the tactic will construct the
`Algebra` property by giving the `RingHom` property as a term.
2. A lemma (or constructor) proving the `Algebra` property from the `RingHom` property. In this case
it is assumed that the `RingHom` property is the final argument, and that no other explicit argument
is needed. The tactic then constructs the `Algebra` property by applying the lemma or constructor.

Finally, if no argument is provided to the `algebraize` attribute, it is assumed that the tagged
declaration has name `RingHom.Property` and that the corresponding `Algebra` property has name
`Algebra.Property`. The attribute then returns `Algebra.Property` (so assume case 1 above). -/
initialize algebraizeAttr : ParametricAttribute Name ←
  registerParametricAttribute {
    name := `algebraize,
    descr :=
"Tag that lets the `algebraize` tactic know which `Algebra` property corresponds to this `RingHom`
property.",
    getParam := algebraizeGetParam }


/-- Given an expression `f` of type `RingHom A B` where `A` and `B` are commutative semirings,
this function adds the instance `Algebra A B` to the context (if it does not already exist).

This function also requires the type of `f`, given by the parameter `ft`. The reason this is done
(even though `ft` can be inferred from `f`) is to avoid recomputing `ft` in the `algebraize` tactic,
as when `algebraize` calls `addAlgebraInstanceFromRingHom` it has already computed `ft`. -/
def addAlgebraInstanceFromRingHom (f ft : Expr) : TacticM Unit := withMainContext do
  let (_, l) := ft.getAppFnArgs
  -- The type of the corresponding algebra instance
  let alg ← mkAppOptM ``Algebra #[l[0]!, l[1]!, none, none]
  -- If the instance already exists, we do not do anything
  unless (← synthInstance? alg).isSome do
  liftMetaTactic fun mvarid => do
    let nm ← mkFreshBinderNameForTactic `algInst
    let mvar ← mvarid.define nm alg (← mkAppM ``RingHom.toAlgebra #[f])
    let (_, mvar) ← mvar.intro1P
    return [mvar]


/-- Given an expression `g.comp f` which is the composition of two `RingHom`s, this function adds
the instance `IsScalarTower A B C` to the context (if it does not already exist). -/
def addIsScalarTowerInstanceFromRingHomComp (fn : Expr) : TacticM Unit := withMainContext do
  let (_, l) := fn.getAppFnArgs
  let tower ← mkAppOptM ``IsScalarTower #[l[0]!, l[1]!, l[2]!, none, none, none]
  -- If the instance already exists, we do not do anything
  unless (← synthInstance? tower).isSome do
  liftMetaTactic fun mvarid => do
    let nm ← mkFreshBinderNameForTactic `scalarTowerInst
    let h ← mkFreshExprMVar (← mkAppM ``Eq #[
      ← mkAppOptM ``algebraMap #[l[0]!, l[2]!, none, none, none],
      ← mkAppM ``RingHom.comp #[
        ← mkAppOptM ``algebraMap #[l[1]!, l[2]!, none, none, none],
        ← mkAppOptM ``algebraMap #[l[0]!, l[1]!, none, none, none]]])
    -- Note: this could fail, but then `algebraize` will just continue, and won't add this instance
    h.mvarId!.refl
    let val ← mkAppOptM ``IsScalarTower.of_algebraMap_eq'
      #[l[0]!, l[1]!, l[2]!, none, none, none, none, none, none, h]
    let mvar ← mvarid.define nm tower val
    let (_, mvar) ← mvar.intro1P
    return [mvar]


/-- This function takes an array of expressions `t`, all of which are assumed to be `RingHom`s,
and searches through the local context to find any additional properties of these `RingHoms`, after
which it tries to add the corresponding `Algebra` properties to the context. It only looks for
properties that have been tagged with the `algebraize` attribute, and uses this tag to find the
corresponding `Algebra` property. -/
def addProperties (t : Array Expr) : TacticM Unit := withMainContext do
  let ctx ← getLCtx
  ctx.forM fun decl => do
    if decl.isImplementationDetail then return
    let (nm, args) := decl.type.getAppFnArgs
    -- Check if the type of the current hypothesis has been tagged with the `algebraize` attribute
    match Attr.algebraizeAttr.getParam? (← getEnv) nm with
    -- If it has, `p` will either be the name of the corresponding `Algebra` property, or a
    -- lemma/constructor.
    | some p =>
      -- The last argument of the `RingHom` property is assumed to be `f`
      let f := args[args.size - 1]!
      -- Check that `f` appears in the list of functions given to `algebraize`
      if ¬ (← t.anyM (Meta.isDefEq · f)) then return

      let cinfo ← getConstInfo p
      let n ← getExpectedNumArgs cinfo.type
      let pargs := Array.mkArray n (none : Option Expr)
      /- If the attribute points to the corresponding `Algebra` property itself, we assume that it
      is definitionally the same as the `RingHom` property. Then, we just need to construct its type
      and the local declaration will already give a valid term. -/
      if cinfo.isInductive then
        let pargs := pargs.set! 0 args[0]!
        let pargs := pargs.set! 1 args[1]!
        let tp ← mkAppOptM p pargs -- This should be the type `Algebra.Property A B`
        unless (← synthInstance? tp).isSome do
        liftMetaTactic fun mvarid => do
          let nm ← mkFreshBinderNameForTactic `algebraizeInst
          let (_, mvar) ← mvarid.note nm decl.toExpr tp
          return [mvar]
      /- Otherwise, the attribute points to a lemma or a constructor for the `Algebra` property.
      In this case, we assume that the `RingHom` property is the last argument of the lemma or
      constructor (and that this is all we need to supply explicitly). -/
      else
        let pargs := pargs.set! (n - 1) decl.toExpr
        let val ← mkAppOptM p pargs
        let tp ← inferType val
        unless (← synthInstance? tp).isSome do
        liftMetaTactic fun mvarid => do
          let nm ← mkFreshBinderNameForTactic `algebraizeInst
          let (_, mvar) ← mvarid.note nm val
          return [mvar]
    | none => return


/-- Configuration for `algebraize`. -/
structure Config where
  /-- If true (default), the tactic will search the local context for `RingHom` properties
    that can be converted to `Algebra` properties. -/
  properties : Bool := true
deriving Inhabited


/-- Function elaborating `Algebraize.Config`. -/
declare_config_elab elabAlgebraizeConfig Algebraize.Config


/-- A list of terms passed to `algebraize` as argument. -/
syntax algebraizeTermSeq := " [" withoutPosition(term,*,?) "]"


/-- Tactic that, given `RingHom`s, adds the corresponding `Algebra` and (if possible)
`IsScalarTower` instances, as well as `Algebra` corresponding to `RingHom` properties available
as hypotheses.

Example: given `f : A →+* B` and `g : B →+* C`, and `hf : f.FiniteType`, `algebraize [f, g]` will
add the instances `Algebra A B`, `Algebra B C`, and `Algebra.FiniteType A B`.

See the `algebraize` tag for instructions on what properties can be added.

The tactic also comes with a configuration option `properties`. If set to `true` (default), the
tactic searches through the local context for `RingHom` properties that can be converted to
`Algebra` properties. The macro `algebraize_only` calls
`algebraize (config := {properties := false})`,
so in other words it only adds `Algebra` and `IsScalarTower` instances. -/
syntax "algebraize " optConfig (algebraizeTermSeq)? : tactic


elab_rules : tactic
  | `(tactic| algebraize $cfg:optConfig $args) => withMainContext do
    let cfg ← elabAlgebraizeConfig cfg
    let t ← match args with
    | `(algebraizeTermSeq| [$rs,*]) => rs.getElems.mapM fun i => Term.elabTerm i none
    | _ =>
      throwError ""
    if t.size == 0 then
      logWarningAt args "`algebraize []` without arguments has no effect!"
    -- We loop through the given terms and add algebra instances
    for f in t do
      let ft ← inferType f
      match ft.getAppFn with
      | Expr.const ``RingHom _ => addAlgebraInstanceFromRingHom f ft
      | _ => throwError m!"{f} is not of type `RingHom`"
    -- After having added the algebra instances we try to add scalar tower instances
    for f in t do
      match f.getAppFn with
      | Expr.const ``RingHom.comp _ =>
        try addIsScalarTowerInstanceFromRingHomComp f
        catch _ => continue
      | _ => continue

    -- Search through the local context to find other instances of algebraize
    if cfg.properties then
      addProperties t
  | `(tactic| algebraize $[$config]?) => do
    throwError "`algebraize` expects a list of arguments: `algebraize [f]`"


/-- Version of `algebraize`, which only adds `Algebra` instances and `IsScalarTower` instances,
but does not try to add any instances about any properties tagged with
`@[algebraize]`, like for example `Finite` or `IsIntegral`. -/
syntax "algebraize_only" (ppSpace algebraizeTermSeq)? : tactic


macro_rules
  | `(tactic| algebraize_only $[$args]?) =>
    `(tactic| algebraize -properties $[$args]?)


